"""User subscription model — Firestore `users/{uid}` document.

Tier lifecycle:
    signup → trial (7 days) → expired → (subscribes) → basic/pro/premium
    admin grant → insider (never expires)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

# Tier constants — ordered by access level
TIER_TRIAL = "trial"
TIER_BASIC = "basic"
TIER_PRO = "pro"
TIER_PREMIUM = "premium"
TIER_INSIDER = "insider"
TIER_EXPIRED = "expired"

ALL_TIERS = (TIER_EXPIRED, TIER_TRIAL, TIER_BASIC, TIER_PRO, TIER_PREMIUM, TIER_INSIDER)

TRIAL_DURATION_DAYS = 7


@dataclass
class UserDoc:
    """Represents a `users/{uid}` Firestore document."""

    uid: str
    tier: str = TIER_TRIAL
    trial_started_at: str | None = None
    trial_expires_at: str | None = None
    tier_expires_at: str | None = None
    platform: str | None = None  # "ios" | "android"
    product_id: str | None = None
    original_transaction_id: str | None = None
    receipt_validated_at: str | None = None
    insider_granted_by: str | None = None  # admin UID who granted insider
    promo_code_redeemed: str | None = None  # code used to upgrade tier
    notifications_enabled: bool | None = None  # None = never set by user
    preferred_language: str | None = None  # IETF tag e.g. "en", "zh-Hant-TW"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def effective_tier(self) -> str:
        """Compute the effective tier accounting for expiration.

        - Insider tier never expires.
        - Trial expires after TRIAL_DURATION_DAYS.
        - Paid tiers expire at tier_expires_at.
        """
        if self.tier == TIER_INSIDER:
            return TIER_INSIDER

        now = datetime.now(timezone.utc)

        if self.tier == TIER_TRIAL:
            if self.trial_expires_at:
                expires = datetime.fromisoformat(self.trial_expires_at)
                if now >= expires:
                    return TIER_EXPIRED
            return TIER_TRIAL

        if self.tier in (TIER_BASIC, TIER_PRO, TIER_PREMIUM):
            if self.tier_expires_at:
                expires = datetime.fromisoformat(self.tier_expires_at)
                if now >= expires:
                    return TIER_EXPIRED
            return self.tier

        return TIER_EXPIRED

    def to_firestore(self) -> dict[str, Any]:
        """Serialize to Firestore document dict."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return {
            "uid": self.uid,
            "tier": self.tier,
            "trial_started_at": self.trial_started_at,
            "trial_expires_at": self.trial_expires_at,
            "tier_expires_at": self.tier_expires_at,
            "platform": self.platform,
            "product_id": self.product_id,
            "original_transaction_id": self.original_transaction_id,
            "receipt_validated_at": self.receipt_validated_at,
            "insider_granted_by": self.insider_granted_by,
            "promo_code_redeemed": self.promo_code_redeemed,
            "notifications_enabled": self.notifications_enabled,
            "preferred_language": self.preferred_language,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_firestore(cls, data: dict[str, Any]) -> UserDoc:
        """Deserialize from Firestore document dict."""
        return cls(
            uid=data.get("uid", ""),
            tier=data.get("tier", TIER_EXPIRED),
            trial_started_at=data.get("trial_started_at"),
            trial_expires_at=data.get("trial_expires_at"),
            tier_expires_at=data.get("tier_expires_at"),
            platform=data.get("platform"),
            product_id=data.get("product_id"),
            original_transaction_id=data.get("original_transaction_id"),
            receipt_validated_at=data.get("receipt_validated_at"),
            insider_granted_by=data.get("insider_granted_by"),
            promo_code_redeemed=data.get("promo_code_redeemed"),
            notifications_enabled=data.get("notifications_enabled"),
            preferred_language=data.get("preferred_language"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    @classmethod
    def create_trial(cls, uid: str, platform: str | None = None) -> UserDoc:
        """Create a new user doc with a 7-day trial."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=TRIAL_DURATION_DAYS)
        return cls(
            uid=uid,
            tier=TIER_TRIAL,
            trial_started_at=now.isoformat(),
            trial_expires_at=expires.isoformat(),
            platform=platform,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
        )
