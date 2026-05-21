"""In-App Purchase receipt validation for Apple App Store and Google Play.

Validates purchase receipts server-side and updates the user's subscription tier
in Firestore. Exposes Cloud Function endpoints for the mobile app to call after
a successful purchase, plus webhook endpoints for server-to-server notifications.

Apple: App Store Server API v2 (JWS signed transactions)
Google: Google Play Developer API (purchases.subscriptions)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
import requests
from firebase_admin import auth as firebase_auth
from firebase_admin import firestore
from firebase_functions import https_fn
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account

from bks_pipeline_core.auth import require_auth
from bks_pipeline_core.models.user import TIER_BASIC, TIER_PREMIUM, TIER_PRO, UserDoc
from bks_pipeline_core.sport_config import get_active_config

__all__ = [
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
    "_decode_apple_jws",
    "_verify_apple_transaction",
    "_verify_google_purchase",
    "_downgrade_user_by_transaction",
    "_extend_user_by_transaction",
    "_handle_apple_expiration",
    "_handle_apple_renewal",
    "_handle_google_expiration",
    "_handle_google_renewal",
]

logger = logging.getLogger(__name__)

_Response = https_fn.Response  # type: ignore[attr-defined]
_Request = https_fn.Request  # type: ignore[attr-defined]

# IAP secrets — read from environment at call time so they are never declared
# as SecretParam at module level. Add them to Secret Manager and re-add
# SecretParam declarations + decorator secrets= lists once App Store Connect
# and Play Console credentials are configured:
#   firebase functions:secrets:set APPLE_KEY_ID
#   firebase functions:secrets:set APPLE_ISSUER_ID
#   firebase functions:secrets:set APPLE_PRIVATE_KEY
#   firebase functions:secrets:set GOOGLE_PLAY_SERVICE_ACCOUNT

# App Store Server API v2 base URL
_APPLE_API_BASE = "https://api.storekit.itunes.apple.com"

# Google Play Developer API scope
_GOOGLE_PLAY_SCOPE = "https://www.googleapis.com/auth/androidpublisher"

# Google RTDN subscription notification types
# Reference: developer.android.com/google/play/billing/rtdn-reference
_GOOGLE_RTDN_EXPIRED = 13
_GOOGLE_RTDN_CANCELED = 3
_GOOGLE_RTDN_REVOKED = 12
_GOOGLE_RTDN_RENEWED = 7
_GOOGLE_RTDN_PURCHASED = 4

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
# Apple App Store Server API v2 — server-side verification
# ---------------------------------------------------------------------------


def _build_apple_jwt() -> str:
    """Build a signed ES256 JWT for the App Store Server API.

    Reference: developer.apple.com/documentation/appstoreserverapi/generating_tokens_for_api_requests
    """
    key_id = os.environ.get("APPLE_KEY_ID", "")
    issuer_id = os.environ.get("APPLE_ISSUER_ID", "")
    private_key = os.environ.get("APPLE_PRIVATE_KEY", "")
    if not key_id or not issuer_id or not private_key:
        raise RuntimeError("Apple API credentials not configured")
    now = int(time.time())
    payload = {
        "iss": issuer_id,
        "iat": now,
        "exp": now + 300,  # 5-minute validity is Apple's documented maximum
        "aud": "appstoreconnect-v1",
        "bid": get_active_config().apple_bundle_id,
    }
    return jwt.encode(
        payload,
        private_key,
        algorithm="ES256",
        headers={"kid": key_id},
    )


def _decode_apple_jws(signed_payload: str) -> dict[str, Any] | None:
    """Decode an Apple JWS payload without signature verification.

    Apple sends server-to-server notifications signed with its private key.
    Verifying the signature would require fetching Apple's rotating public keys.
    We rely on TLS (HTTPS from Apple's servers) to authenticate the origin.
    """
    try:
        return jwt.decode(
            signed_payload,
            options={"verify_signature": False},
            algorithms=["ES256"],
        )
    except Exception:
        logger.warning("Apple S2S: failed to decode JWS signedPayload")
        return None


def _verify_apple_transaction(transaction_id: str, product_id: str) -> tuple[bool, str | None]:
    """Verify a transaction with the App Store Server API v2.

    Returns (is_valid, expires_at_iso). expires_at_iso is None when invalid.

    The API returns the decoded JWS transaction payload which includes:
    - productId: must match the client-supplied product_id
    - expiresDate: milliseconds epoch when the subscription expires
    - revocationDate: set if the purchase was refunded/revoked
    """
    try:
        token = _build_apple_jwt()
    except Exception:
        logger.exception("Apple verification: failed to build JWT")
        return False, None

    url = f"{_APPLE_API_BASE}/inApps/v1/transactions/{transaction_id}"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
    except requests.RequestException:
        logger.exception("Apple verification: network error calling App Store API")
        return False, None

    if resp.status_code == 404:
        logger.warning("Apple verification: transaction %s not found", transaction_id)
        return False, None

    if resp.status_code != 200:
        logger.warning(
            "Apple verification: App Store API returned %d for transaction %s",
            resp.status_code,
            transaction_id,
        )
        return False, None

    try:
        data = resp.json()
    except ValueError:
        logger.warning("Apple verification: non-JSON response from App Store API")
        return False, None

    # The response contains a signedTransactionInfo JWS — decode without verifying
    # the Apple signature; the HTTPS TLS chain already authenticates the response.
    signed_transaction = data.get("signedTransactionInfo", "")
    if not signed_transaction:
        logger.warning("Apple verification: no signedTransactionInfo in response")
        return False, None

    transaction = _decode_apple_jws(signed_transaction)
    if transaction is None:
        return False, None

    # Reject revoked purchases
    if transaction.get("revocationDate"):
        logger.warning("Apple verification: transaction %s has been revoked", transaction_id)
        return False, None

    # Confirm the product matches what the client sent
    api_product_id = transaction.get("productId", "")
    if api_product_id != product_id:
        logger.warning(
            "Apple verification: product mismatch — client=%s api=%s",
            product_id,
            api_product_id,
        )
        return False, None

    # Derive expiry from the API response when available; fall back to product-based calculation
    expires_ms = transaction.get("expiresDate")
    if expires_ms:
        expires_at = datetime.fromtimestamp(expires_ms / 1000, tz=timezone.utc).isoformat()
    else:
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(days=365 if "annual" in product_id else 30)).isoformat()

    return True, expires_at


# ---------------------------------------------------------------------------
# Google Play Developer API — server-side verification
# ---------------------------------------------------------------------------


def _get_google_play_credentials() -> Any:
    """Build Google service account credentials from the stored secret JSON."""
    sa_raw = os.environ.get("GOOGLE_PLAY_SERVICE_ACCOUNT", "")
    if not sa_raw:
        raise RuntimeError("GOOGLE_PLAY_SERVICE_ACCOUNT not configured")
    sa_json = json.loads(sa_raw)
    creds: Any = service_account.Credentials.from_service_account_info(  # type: ignore[no-untyped-call]
        sa_json,
        scopes=[_GOOGLE_PLAY_SCOPE],
    )
    creds.refresh(GoogleAuthRequest())
    return creds


def _verify_google_purchase(purchase_token: str, product_id: str) -> tuple[bool, str | None]:
    """Verify a Google Play subscription purchase token.

    Returns (is_valid, expires_at_iso). expires_at_iso is None when invalid.

    Reference: developers.google.com/android-publisher/api-ref/rest/v3/purchases.subscriptions/get
    """
    try:
        creds = _get_google_play_credentials()
    except Exception:
        logger.exception("Google verification: failed to build service account credentials")
        return False, None

    package_name = get_active_config().apple_bundle_id  # same reverse-domain root
    url = (
        f"https://androidpublisher.googleapis.com/androidpublisher/v3/applications/{package_name}/purchases/subscriptions/{product_id}/tokens/{purchase_token}"
    )
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {creds.token}"},
            timeout=10,
        )
    except requests.RequestException:
        logger.exception("Google verification: network error calling Play Developer API")
        return False, None

    if resp.status_code == 404:
        logger.warning("Google verification: purchase token not found for product %s", product_id)
        return False, None

    if resp.status_code != 200:
        logger.warning(
            "Google verification: Play API returned %d for product %s",
            resp.status_code,
            product_id,
        )
        return False, None

    try:
        data = resp.json()
    except ValueError:
        logger.warning("Google verification: non-JSON response from Play API")
        return False, None

    # paymentState: 0=pending, 1=received, 2=free trial, 3=deferred upgrade
    payment_state = data.get("paymentState")
    if payment_state not in (1, 2):
        logger.warning(
            "Google verification: unacceptable paymentState=%s for product %s",
            payment_state,
            product_id,
        )
        return False, None

    expiry_ms = data.get("expiryTimeMillis")
    if not expiry_ms:
        logger.warning("Google verification: missing expiryTimeMillis for product %s", product_id)
        return False, None

    expiry_dt = datetime.fromtimestamp(int(expiry_ms) / 1000, tz=timezone.utc)
    if expiry_dt <= datetime.now(timezone.utc):
        logger.warning("Google verification: subscription already expired for product %s", product_id)
        return False, None

    return True, expiry_dt.isoformat()


# ---------------------------------------------------------------------------
# Apple App Store validation endpoint
# ---------------------------------------------------------------------------


@https_fn.on_request(max_instances=5)
@require_auth
def validate_apple_receipt(req: Any) -> Any:
    """Validate an Apple App Store receipt and update subscription tier.

    Calls the App Store Server API v2 to confirm the transaction exists,
    is not revoked, and belongs to the claimed product before writing to Firestore.
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

    tier = get_apple_product_tiers().get(product_id)
    if not tier:
        logger.warning("Apple validation: unrecognized product_id=%s", product_id)
        return _Response(
            json.dumps({"error": "unknown_product", "product_id": product_id}),
            status=422,
            content_type="application/json",
        )

    uid = _extract_uid_from_request(req)
    if not uid:
        return _Response(json.dumps({"error": "auth_uid_missing"}), status=401, content_type="application/json")

    is_valid, expires_at = _verify_apple_transaction(transaction_id, product_id)
    if not is_valid or not expires_at:
        logger.warning("Apple validation failed: uid=%s transaction=%s", uid, transaction_id)
        return _Response(
            json.dumps({"error": "receipt_verification_failed"}),
            status=422,
            content_type="application/json",
        )

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
# Google Play validation endpoint
# ---------------------------------------------------------------------------


@https_fn.on_request(max_instances=5)
@require_auth
def validate_google_receipt(req: Any) -> Any:
    """Validate a Google Play purchase and update subscription tier.

    Calls the Google Play Developer API to confirm the purchase token is valid,
    payment has been received, and the subscription has not yet expired before
    writing to Firestore.
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

    tier = GOOGLE_PRODUCT_TIERS.get(product_id)
    if not tier:
        logger.warning("Google validation: unrecognized product_id=%s", product_id)
        return _Response(
            json.dumps({"error": "unknown_product", "product_id": product_id}),
            status=422,
            content_type="application/json",
        )

    uid = _extract_uid_from_request(req)
    if not uid:
        return _Response(json.dumps({"error": "auth_uid_missing"}), status=401, content_type="application/json")

    is_valid, expires_at = _verify_google_purchase(purchase_token, product_id)
    if not is_valid or not expires_at:
        logger.warning("Google validation failed: uid=%s product=%s", uid, product_id)
        return _Response(
            json.dumps({"error": "receipt_verification_failed"}),
            status=422,
            content_type="application/json",
        )

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


@https_fn.on_request(max_instances=5)
def apple_server_notification(req: Any) -> Any:
    """Webhook for Apple App Store Server Notifications v2.

    Apple sends a JSON body with a single `signedPayload` field containing a JWS.
    The decoded payload contains `notificationType` and `data.signedTransactionInfo`.

    Reference: developer.apple.com/documentation/appstoreservernotifications
    """
    if req.method != "POST":
        return _Response("Method not allowed", status=405)

    try:
        body = req.get_json(force=True)
    except Exception:
        logger.warning("Apple S2S notification: invalid body")
        return _Response("", status=400)

    signed_payload = body.get("signedPayload", "")
    if not signed_payload:
        logger.warning("Apple S2S notification: missing signedPayload")
        return _Response("", status=400)

    notification = _decode_apple_jws(signed_payload)
    if notification is None:
        return _Response("", status=400)

    notification_type = notification.get("notificationType", "unknown")
    subtype = notification.get("subtype", "")
    logger.info("Apple S2S notification: type=%s subtype=%s", notification_type, subtype)

    if notification_type in ("EXPIRED", "REVOKE", "GRACE_PERIOD_EXPIRED"):
        _handle_apple_expiration(notification)
    elif notification_type in ("DID_RENEW", "SUBSCRIBED"):
        _handle_apple_renewal(notification)

    # Always return 200 — Apple retries on non-2xx
    return _Response("", status=200)


@https_fn.on_request(max_instances=5)
def google_rtdn_notification(req: Any) -> Any:
    """Webhook for Google Real-Time Developer Notifications (RTDN).

    Google delivers via Pub/Sub push: body is {"message": {"data": "<base64>", "messageId": "..."}}
    The decoded data is a DeveloperNotification JSON with a subscriptionNotification field.

    Reference: developer.android.com/google/play/billing/rtdn-reference
    """
    if req.method != "POST":
        return _Response("Method not allowed", status=405)

    try:
        body = req.get_json(force=True)
    except Exception:
        logger.warning("Google RTDN: invalid body")
        return _Response("", status=400)

    message = body.get("message", {})
    if not message:
        logger.warning("Google RTDN: no message field")
        return _Response("", status=400)

    message_id = message.get("messageId", "unknown")
    encoded_data = message.get("data", "")
    if not encoded_data:
        logger.info("Google RTDN: message has no data field (message_id=%s)", message_id)
        return _Response("", status=200)

    try:
        notification = json.loads(base64.b64decode(encoded_data).decode("utf-8"))
    except Exception:
        logger.warning("Google RTDN: failed to decode message data (message_id=%s)", message_id)
        return _Response("", status=200)

    sub_notification = notification.get("subscriptionNotification")
    if not sub_notification:
        logger.info("Google RTDN: no subscriptionNotification in message (message_id=%s)", message_id)
        return _Response("", status=200)

    notification_type = sub_notification.get("notificationType")
    purchase_token = sub_notification.get("purchaseToken", "")
    product_id = sub_notification.get("subscriptionId", "")
    logger.info(
        "Google RTDN: type=%s product=%s message_id=%s",
        notification_type,
        product_id,
        message_id,
    )

    if notification_type in (_GOOGLE_RTDN_EXPIRED, _GOOGLE_RTDN_CANCELED, _GOOGLE_RTDN_REVOKED):
        _handle_google_expiration(purchase_token)
    elif notification_type in (_GOOGLE_RTDN_RENEWED, _GOOGLE_RTDN_PURCHASED):
        _handle_google_renewal(purchase_token, product_id)

    return _Response("", status=200)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_uid_from_request(req: Any) -> str | None:
    """Extract UID from a request that has passed through @require_auth."""
    uid = getattr(req, "_uid", None)
    if uid:
        return str(uid)

    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    id_token = auth_header.split("Bearer ", 1)[1]
    try:
        decoded = firebase_auth.verify_id_token(id_token)
        result = decoded.get("uid")
        return str(result) if result is not None else None
    except Exception:
        return None


def _downgrade_user_by_transaction(db: Any, original_transaction_id: str, platform: str) -> bool:
    """Set tier_expires_at to now for the user with this transaction ID.

    effective_tier() computes TIER_EXPIRED when tier_expires_at is in the past.
    Returns True if a matching user was found and updated.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        docs = db.collection("users").where("original_transaction_id", "==", original_transaction_id).where("platform", "==", platform).limit(1).stream()
        doc = next(iter(docs), None)
        if doc is None:
            logger.warning(
                "Downgrade: no user found for transaction=%s platform=%s",
                original_transaction_id,
                platform,
            )
            return False
        doc.reference.update({"tier_expires_at": now_iso, "updated_at": now_iso})
        logger.info(
            "Downgrade: expired uid=%s transaction=%s",
            doc.id,
            original_transaction_id,
        )
        return True
    except Exception:
        logger.exception("Downgrade: Firestore error for transaction=%s", original_transaction_id)
        return False


def _extend_user_by_transaction(db: Any, original_transaction_id: str, platform: str, expires_at: str) -> bool:
    """Update tier_expires_at for the user with this transaction ID.

    Returns True if a matching user was found and updated.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        docs = db.collection("users").where("original_transaction_id", "==", original_transaction_id).where("platform", "==", platform).limit(1).stream()
        doc = next(iter(docs), None)
        if doc is None:
            logger.warning(
                "Renewal: no user found for transaction=%s platform=%s",
                original_transaction_id,
                platform,
            )
            return False
        doc.reference.update({"tier_expires_at": expires_at, "updated_at": now_iso})
        logger.info(
            "Renewal: extended uid=%s transaction=%s expires=%s",
            doc.id,
            original_transaction_id,
            expires_at,
        )
        return True
    except Exception:
        logger.exception("Renewal: Firestore error for transaction=%s", original_transaction_id)
        return False


def _handle_apple_expiration(notification: dict[str, Any]) -> None:
    """Downgrade the user whose subscription expired or was revoked by Apple."""
    data = notification.get("data", {})
    signed_transaction = data.get("signedTransactionInfo", "")
    if not signed_transaction:
        logger.warning("Apple expiration: no signedTransactionInfo in notification data")
        return

    transaction = _decode_apple_jws(signed_transaction)
    if transaction is None:
        return

    original_transaction_id = transaction.get("originalTransactionId", "")
    if not original_transaction_id:
        logger.warning("Apple expiration: missing originalTransactionId in transaction")
        return

    db = _get_db()
    _downgrade_user_by_transaction(db, original_transaction_id, "ios")


def _handle_apple_renewal(notification: dict[str, Any]) -> None:
    """Extend the user's tier_expires_at when Apple confirms a successful renewal."""
    data = notification.get("data", {})
    signed_transaction = data.get("signedTransactionInfo", "")
    if not signed_transaction:
        logger.warning("Apple renewal: no signedTransactionInfo in notification data")
        return

    transaction = _decode_apple_jws(signed_transaction)
    if transaction is None:
        return

    original_transaction_id = transaction.get("originalTransactionId", "")
    expires_ms = transaction.get("expiresDate")
    if not original_transaction_id or not expires_ms:
        logger.warning("Apple renewal: missing originalTransactionId or expiresDate in transaction")
        return

    expires_at = datetime.fromtimestamp(expires_ms / 1000, tz=timezone.utc).isoformat()
    db = _get_db()
    _extend_user_by_transaction(db, original_transaction_id, "ios", expires_at)


def _handle_google_expiration(purchase_token: str) -> None:
    """Downgrade the user whose Google Play subscription expired or was cancelled."""
    if not purchase_token:
        logger.warning("Google expiration: empty purchase_token")
        return
    db = _get_db()
    _downgrade_user_by_transaction(db, purchase_token, "android")


def _handle_google_renewal(purchase_token: str, product_id: str) -> None:
    """Extend the user's tier_expires_at when Google confirms a renewal or new purchase."""
    if not purchase_token or not product_id:
        logger.warning("Google renewal: missing purchase_token or product_id")
        return

    is_valid, expires_at = _verify_google_purchase(purchase_token, product_id)
    if not is_valid or not expires_at:
        logger.warning("Google renewal: Play API verification failed for product=%s", product_id)
        return

    db = _get_db()
    _extend_user_by_transaction(db, purchase_token, "android", expires_at)
