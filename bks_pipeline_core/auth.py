import functools
import json
import logging
from typing import Any, Callable

from firebase_admin import auth, firestore
from firebase_functions import https_fn

from bks_pipeline_core.models.user import TIER_EXPIRED, TIER_TRIAL, UserDoc
from bks_pipeline_core.sport_config import get_active_config

# firebase_functions stubs don't export Response; access at runtime only.
_Response = https_fn.Response  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


def _get_or_create_user(uid: str) -> UserDoc:
    """Read or lazily create a user doc in Firestore.

    On first authenticated request, creates a trial user. On subsequent
    requests, reads the existing doc. If Firestore read fails, returns a
    trial-tier user (graceful degradation — never hard-block due to infra error).
    """
    try:
        db = firestore.client(database_id=get_active_config().firestore_database_id)
        doc_ref = db.collection("users").document(uid)
        doc = doc_ref.get()
        if doc.exists:
            return UserDoc.from_firestore(doc.to_dict())
        # First request — create trial user
        user = UserDoc.create_trial(uid)
        doc_ref.set(user.to_firestore())
        logger.info("Created trial user doc: uid=%s", uid)
        return user
    except Exception:
        logger.exception("Failed to read/create user doc for uid=%s, defaulting to trial", uid)
        return UserDoc(uid=uid, tier=TIER_TRIAL)


def require_admin(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that validates a Firebase ID token AND requires the 'admin' custom claim.

    Returns 401 if the Authorization: Bearer <token> header is missing or the token is invalid.
    Returns 403 if the token is valid but the user does not have the 'admin' custom claim.
    Returns 503 if token verification fails due to an unexpected error.

    To grant admin access to a user, set the custom claim via Admin SDK:
        auth.set_custom_user_claims(uid, {"admin": True})
    """

    @functools.wraps(func)
    def wrapper(req: Any, *args: Any, **kwargs: Any) -> Any:
        auth_header = req.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return _Response("Unauthorized", status=401)
        id_token = auth_header.split("Bearer ", 1)[1]
        try:
            decoded = auth.verify_id_token(id_token)
        except auth.ExpiredIdTokenError:
            logger.warning("Admin auth rejected: token expired")
            return _Response("Unauthorized", status=401)
        except auth.RevokedIdTokenError:
            logger.warning("Admin auth rejected: token revoked")
            return _Response("Unauthorized", status=401)
        except auth.UserDisabledError:
            logger.warning("Admin auth rejected: user disabled")
            return _Response("Unauthorized", status=401)
        except (auth.InvalidIdTokenError, auth.CertificateFetchError):
            return _Response("Unauthorized", status=401)
        except Exception:
            logger.exception("Unexpected error during admin token verification")
            return _Response("Service unavailable", status=503)
        if not decoded.get("admin"):
            logger.warning("Admin auth rejected: uid=%s lacks admin claim", decoded.get("uid"))
            return _Response("Forbidden", status=403)
        req._uid = decoded.get("uid")
        req._tier = "insider"  # Admins always have insider-level access
        logger.info("Admin auth OK: uid=%s path=%s", decoded.get("uid"), req.path)
        return func(req, *args, **kwargs)

    return wrapper


def require_auth(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that validates a Firebase ID token before calling the wrapped function.

    After successful authentication:
    - Attaches ``req._uid`` (str) — the user's Firebase UID
    - Attaches ``req._tier`` (str) — the user's effective subscription tier

    The user doc is lazily created on first request (7-day trial).

    Returns 401 if the Authorization: Bearer <token> header is missing or the token is invalid.
    Returns 503 if token verification fails due to an unexpected error (e.g. Firebase Auth outage).
    """

    @functools.wraps(func)
    def wrapper(req: Any, *args: Any, **kwargs: Any) -> Any:
        auth_header = req.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return _Response("Unauthorized", status=401)
        id_token = auth_header.split("Bearer ", 1)[1]
        try:
            decoded = auth.verify_id_token(id_token)
        except auth.ExpiredIdTokenError:
            logger.warning("Auth rejected: token expired")
            return _Response("Unauthorized", status=401)
        except auth.RevokedIdTokenError:
            logger.warning("Auth rejected: token revoked")
            return _Response("Unauthorized", status=401)
        except auth.UserDisabledError:
            logger.warning("Auth rejected: user disabled")
            return _Response("Unauthorized", status=401)
        except (auth.InvalidIdTokenError, auth.CertificateFetchError):
            return _Response("Unauthorized", status=401)
        except Exception:
            logger.exception("Unexpected error during token verification")
            return _Response("Service unavailable", status=503)

        uid = decoded.get("uid", "")
        user_doc = _get_or_create_user(uid)
        effective_tier = user_doc.effective_tier()

        req._uid = uid
        req._tier = effective_tier
        logger.info("Auth OK: uid=%s tier=%s path=%s", uid, effective_tier, req.path)
        return func(req, *args, **kwargs)

    return wrapper


def require_tier(*allowed_tiers: str) -> Callable[..., Any]:
    """Decorator that enforces minimum subscription tier access.

    Must be applied AFTER @require_auth (so req._tier is available).

    Usage:
        @https_fn.on_request()
        @require_auth
        @require_tier("pro", "premium", "insider")
        def my_endpoint(req):
            ...

    Returns 402 if the user's tier has expired (subscription_required).
    Returns 403 if the user's tier is not in the allowed list.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(req: Any, *args: Any, **kwargs: Any) -> Any:
            tier = getattr(req, "_tier", TIER_EXPIRED)

            if tier == TIER_EXPIRED:
                return _Response(
                    json.dumps({"error": "subscription_required", "trial_expired": True}),
                    status=402,
                    content_type="application/json",
                )

            if tier not in allowed_tiers:
                return _Response(
                    json.dumps(
                        {
                            "error": "upgrade_required",
                            "current_tier": tier,
                            "required_tiers": list(allowed_tiers),
                        }
                    ),
                    status=403,
                    content_type="application/json",
                )

            return func(req, *args, **kwargs)

        return wrapper

    return decorator
