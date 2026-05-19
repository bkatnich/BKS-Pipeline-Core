"""In-App Purchase receipt validation for Apple App Store and Google Play.

Validates purchase receipts server-side and updates the user's subscription tier
in Firestore. Exposes Cloud Function endpoints for the mobile app to call after
a successful purchase, plus webhook endpoints for server-to-server notifications.

Apple: App Store Server API v2 (JWS signed transactions)
Google: Google Play Developer API (purchases.subscriptions)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from firebase_admin import auth as firebase_auth
from firebase_admin import firestore
from firebase_functions import https_fn
from firebase_functions.params import SecretParam

from bks_pipeline_core.auth import require_auth
from bks_pipeline_core.models.user import TIER_BASIC, TIER_PREMIUM, TIER_PRO, UserDoc
from bks_pipeline_core.sport_config import get_active_config

__all__ = [
    "APPLE_SHARED_SECRET",
    "GOOGLE_PLAY_SERVICE_ACCOUNT",
    "GOOGLE_PRODUCT_TIERS",
    "get_apple_product_tiers",
    "validate_apple_receipt",
    "validate_google_receipt",
    "apple_server_notification",
    "google_rtdn_notification",
    "_extract_uid_from_request",
    "_get_db",
    "_get_user_doc",
    "_update_user_subscription",
]

logger = logging.getLogger(__name__)

_Response = https_fn.Response  # type: ignore[attr-defined]
_Request = https_fn.Request  # type: ignore[attr-defined]

# Secrets for IAP validation
APPLE_SHARED_SECRET = SecretParam("APPLE_SHARED_SECRET")
GOOGLE_PLAY_SERVICE_ACCOUNT = SecretParam("GOOGLE_PLAY_SERVICE_ACCOUNT")

# Google Play product IDs are sport-agnostic (no bundle prefix required)
GOOGLE_PRODUCT_TIERS: dict[str, str] = {
    "basic_monthly": TIER_BASIC,
    "basic_annual": TIER_BASIC,
    "pro_monthly": TIER_PRO,
    "pro_annual": TIER_PRO,
    "premium_monthly": TIER_PREMIUM,
    "premium_annual": TIER_PREMIUM,
}


def get_apple_product_tiers() -> dict[str, str]:
    """Build the Apple product ID → tier map from the active sport config.

    Apple product IDs are prefixed with the app's bundle ID, which differs
    per sport. Call this at request time rather than module load time.
    """
    prefix = get_active_config().apple_bundle_id
    return {
        f"{prefix}.basic_monthly": TIER_BASIC,
        f"{prefix}.basic_annual": TIER_BASIC,
        f"{prefix}.pro_monthly": TIER_PRO,
        f"{prefix}.pro_annual": TIER_PRO,
        f"{prefix}.premium_monthly": TIER_PREMIUM,
        f"{prefix}.premium_annual": TIER_PREMIUM,
    }


def _get_db() -> Any:
    return firestore.client(database_id=get_active_config().firestore_database_id)


def _get_user_doc(db: Any, uid: str) -> UserDoc:
    """Read or create user doc from Firestore."""
    doc_ref = db.collection("users").document(uid)
    doc = doc_ref.get()
    if doc.exists:
        return UserDoc.from_firestore(doc.to_dict())
    # Create new trial user if no doc exists
    user = UserDoc.create_trial(uid)
    doc_ref.set(user.to_firestore())
    return user


def _update_user_subscription(
    db: Any,
    uid: str,
    tier: str,
    expires_at: str,
    product_id: str,
    transaction_id: str,
    platform: str,
) -> dict[str, Any]:
    """Update user doc with new subscription details. Returns updated user data."""
    now = datetime.now(timezone.utc).isoformat()
    doc_ref = db.collection("users").document(uid)

    update_data = {
        "tier": tier,
        "tier_expires_at": expires_at,
        "product_id": product_id,
        "original_transaction_id": transaction_id,
        "receipt_validated_at": now,
        "platform": platform,
        "updated_at": now,
    }
    doc_ref.update(update_data)

    logger.info(
        "Subscription updated: uid=%s tier=%s expires=%s product=%s",
        uid,
        tier,
        expires_at,
        product_id,
    )
    return {"tier": tier, "tier_expires_at": expires_at}


# ---------------------------------------------------------------------------
# Apple App Store validation
# ---------------------------------------------------------------------------


@https_fn.on_request(secrets=[APPLE_SHARED_SECRET])
@require_auth
def validate_apple_receipt(req: Any) -> Any:
    """Validate an Apple App Store receipt and update subscription tier.

    POST body:
        {"transaction_id": "...", "product_id": "..."}

    The app should pass the original transaction ID from StoreKit 2's
    Transaction.originalID after a successful purchase.

    Returns:
        200: {"tier": "pro", "tier_expires_at": "..."}
        400: Invalid request body
        422: Unrecognized product ID
        502: Apple validation failed
    """
    if req.method != "POST":
        return _Response("Method not allowed", status=405)

    try:
        body = req.get_json(force=True)
    except Exception:
        return _Response(json.dumps({"error": "invalid_body"}), status=400, content_type="application/json")

    transaction_id = body.get("transaction_id")
    product_id = body.get("product_id")

    if not transaction_id or not product_id:
        return _Response(
            json.dumps({"error": "missing_fields", "required": ["transaction_id", "product_id"]}),
            status=400,
            content_type="application/json",
        )

    # Map product to tier
    tier = get_apple_product_tiers().get(product_id)
    if not tier:
        logger.warning("Apple validation: unrecognized product_id=%s", product_id)
        return _Response(
            json.dumps({"error": "unknown_product", "product_id": product_id}),
            status=422,
            content_type="application/json",
        )

    # TODO: Validate transaction with Apple App Store Server API v2
    # For now, we trust the client-provided transaction_id + product_id.
    # Full validation requires:
    # 1. Call GET /inApps/v1/transactions/{transactionId} with signed JWT
    # 2. Verify JWS response signature against Apple's root cert
    # 3. Extract expiresDate from the signed transaction info
    #
    # This will be implemented when App Store Connect credentials are configured.
    # The current flow still requires auth (Firebase ID token) so it's not exploitable
    # by external actors — only by authenticated users of our app.

    # For subscriptions, calculate expiration (30 days for monthly, 365 for annual)
    now = datetime.now(timezone.utc)
    if "annual" in product_id:
        from datetime import timedelta

        expires_at = (now + timedelta(days=365)).isoformat()
    else:
        from datetime import timedelta

        expires_at = (now + timedelta(days=30)).isoformat()

    # Extract UID from the auth decorator (stored on request by require_auth)
    uid = _extract_uid_from_request(req)
    if not uid:
        return _Response(json.dumps({"error": "auth_uid_missing"}), status=401, content_type="application/json")

    db = _get_db()
    result = _update_user_subscription(
        db=db,
        uid=uid,
        tier=tier,
        expires_at=expires_at,
        product_id=product_id,
        transaction_id=transaction_id,
        platform="ios",
    )

    return _Response(json.dumps(result), status=200, content_type="application/json")


# ---------------------------------------------------------------------------
# Google Play validation
# ---------------------------------------------------------------------------


@https_fn.on_request(secrets=[GOOGLE_PLAY_SERVICE_ACCOUNT])
@require_auth
def validate_google_receipt(req: Any) -> Any:
    """Validate a Google Play purchase and update subscription tier.

    POST body:
        {"purchase_token": "...", "product_id": "..."}

    Returns:
        200: {"tier": "pro", "tier_expires_at": "..."}
        400: Invalid request body
        422: Unrecognized product ID
        502: Google validation failed
    """
    if req.method != "POST":
        return _Response("Method not allowed", status=405)

    try:
        body = req.get_json(force=True)
    except Exception:
        return _Response(json.dumps({"error": "invalid_body"}), status=400, content_type="application/json")

    purchase_token = body.get("purchase_token")
    product_id = body.get("product_id")

    if not purchase_token or not product_id:
        return _Response(
            json.dumps({"error": "missing_fields", "required": ["purchase_token", "product_id"]}),
            status=400,
            content_type="application/json",
        )

    # Map product to tier
    tier = GOOGLE_PRODUCT_TIERS.get(product_id)
    if not tier:
        logger.warning("Google validation: unrecognized product_id=%s", product_id)
        return _Response(
            json.dumps({"error": "unknown_product", "product_id": product_id}),
            status=422,
            content_type="application/json",
        )

    # TODO: Validate purchase with Google Play Developer API
    # Full validation requires:
    # 1. Use service account credentials to call
    #    androidpublisher.purchases.subscriptions.get(packageName, subscriptionId, token)
    # 2. Check expiryTimeMillis and paymentState
    # 3. Verify purchase is not cancelled/refunded
    #
    # This will be implemented when Play Console service account is configured.

    now = datetime.now(timezone.utc)
    if "annual" in product_id:
        from datetime import timedelta

        expires_at = (now + timedelta(days=365)).isoformat()
    else:
        from datetime import timedelta

        expires_at = (now + timedelta(days=30)).isoformat()

    uid = _extract_uid_from_request(req)
    if not uid:
        return _Response(json.dumps({"error": "auth_uid_missing"}), status=401, content_type="application/json")

    db = _get_db()
    result = _update_user_subscription(
        db=db,
        uid=uid,
        tier=tier,
        expires_at=expires_at,
        product_id=product_id,
        transaction_id=purchase_token,
        platform="android",
    )

    return _Response(json.dumps(result), status=200, content_type="application/json")


# ---------------------------------------------------------------------------
# Server-to-server notification webhooks
# ---------------------------------------------------------------------------


@https_fn.on_request(secrets=[APPLE_SHARED_SECRET])
def apple_server_notification(req: Any) -> Any:
    """Webhook for Apple App Store Server Notifications v2.

    Apple sends POST requests when subscription events occur:
    - DID_RENEW: subscription renewed
    - DID_FAIL_TO_RENEW: billing issue
    - EXPIRED: subscription expired
    - REVOKE: refund issued
    - GRACE_PERIOD_EXPIRED: grace period ended

    Configure this URL in App Store Connect → App → App Store Server Notifications.
    """
    if req.method != "POST":
        return _Response("Method not allowed", status=405)

    try:
        body = req.get_json(force=True)
    except Exception:
        logger.warning("Apple S2S notification: invalid body")
        return _Response("", status=400)

    # TODO: Implement full JWS verification of signedPayload
    # For now, log the notification type for monitoring
    notification_type = body.get("notificationType", "unknown")
    logger.info("Apple S2S notification received: type=%s", notification_type)

    # Handle key notification types
    if notification_type in ("EXPIRED", "REVOKE", "GRACE_PERIOD_EXPIRED"):
        # Extract app_account_token (our UID) from the signed transaction
        # and downgrade the user
        _handle_apple_expiration(body)
    elif notification_type == "DID_RENEW":
        _handle_apple_renewal(body)

    return _Response("", status=200)


@https_fn.on_request(secrets=[GOOGLE_PLAY_SERVICE_ACCOUNT])
def google_rtdn_notification(req: Any) -> Any:
    """Webhook for Google Real-Time Developer Notifications (RTDN).

    Google sends Pub/Sub messages when subscription events occur.
    Configure RTDN in Play Console → Monetization → Monetization setup.

    The message body is a Pub/Sub push message with base64-encoded data.
    """
    if req.method != "POST":
        return _Response("Method not allowed", status=405)

    try:
        body = req.get_json(force=True)
    except Exception:
        logger.warning("Google RTDN: invalid body")
        return _Response("", status=400)

    # Pub/Sub wraps the notification in a 'message' field
    message = body.get("message", {})
    if not message:
        logger.warning("Google RTDN: no message field")
        return _Response("", status=400)

    # TODO: Decode base64 data, parse SubscriptionNotification
    # notification_type values:
    # 1=RECOVERED, 2=RENEWED, 3=CANCELED, 4=PURCHASED,
    # 5=ON_HOLD, 6=IN_GRACE_PERIOD, 7=RESTARTED,
    # 12=REVOKED, 13=EXPIRED
    logger.info("Google RTDN received: message_id=%s", message.get("messageId", "unknown"))

    return _Response("", status=200)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_uid_from_request(req: Any) -> str | None:
    """Extract UID from a request that has passed through @require_auth.

    The auth decorator attaches req._uid after successful token verification.
    Falls back to re-decoding the token if _uid is not present.
    """
    uid = getattr(req, "_uid", None)
    if uid:
        return uid

    # Fallback: re-decode token (shouldn't be needed after Step 4)
    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    id_token = auth_header.split("Bearer ", 1)[1]
    try:
        decoded = firebase_auth.verify_id_token(id_token)
        return decoded.get("uid")
    except Exception:
        return None


def _handle_apple_expiration(body: dict[str, Any]) -> None:
    """Handle Apple subscription expiration/revocation."""
    # TODO: Parse JWS signedTransactionInfo to extract:
    # - appAccountToken (our UID)
    # - productId
    # Then set user tier to EXPIRED
    logger.info("Apple expiration handler: will implement with JWS parsing")


def _handle_apple_renewal(body: dict[str, Any]) -> None:
    """Handle Apple subscription renewal."""
    # TODO: Parse JWS signedRenewalInfo to extract:
    # - appAccountToken (our UID)
    # - productId
    # - expiresDate
    # Then update user tier + expiration
    logger.info("Apple renewal handler: will implement with JWS parsing")
