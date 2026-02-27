"""
CSAO Rail Recommendation System — Step 2: Hard Pre-Filters

Deterministic rules applied before any ML model. Run in strict order:
  Filter A — Availability and Margin
  Filter B — Dietary Toggle
  Filter C — Cuisine Coherence
  Filter D — Quantity Saturation
  Filter E — Deduplication and Recommendation Fatigue
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from typing import Any

import pandas as pd

# ────────────────────────────────────────────────────────────────
# DEFAULT SATURATION CAPS (subcategory → max allowed in cart)
# ────────────────────────────────────────────────────────────────
DEFAULT_SATURATION_CAPS: dict[str, int] = {
    "rice": 3,
    "biryani": 2,
    "naan": 4,
    "roti": 4,
    "paratha": 4,
    "soft_drink": 2,
    "lassi": 2,
    "curry": 4,
}
DEFAULT_CAP = 3

MIN_MARGIN_PCT = 10.0
FATIGUE_THRESHOLD = 3
GROUP_ORDER_SIZE = 5


class HardFilterPipeline:
    """Chains five deterministic filters to reduce a restaurant menu
    to a clean candidate pool for downstream ranking."""

    def __init__(
        self,
        menu: pd.DataFrame,
        restaurants: pd.DataFrame,
        users: pd.DataFrame,
        saturation_caps: dict[str, int] | None = None,
    ):
        self.menu = menu
        self.restaurants = restaurants.set_index("restaurant_id")
        self.users = users.set_index("user_id")
        self.saturation_caps = saturation_caps or DEFAULT_SATURATION_CAPS

    # ────────────────────────────────────────────────────────
    # Filter A — Availability and Margin
    # ────────────────────────────────────────────────────────
    def filter_a_availability_margin(
        self, candidates: pd.DataFrame, restaurant_id: str
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Remove out-of-stock and low-margin items; verify promotions."""
        log: dict[str, int] = {"input": len(candidates)}

        mask_avail = candidates["availability"] == True  # noqa: E712
        removed_avail = int((~mask_avail).sum())
        candidates = candidates[mask_avail]
        log["removed_out_of_stock"] = removed_avail

        mask_margin = candidates["margin_pct"] >= MIN_MARGIN_PCT
        removed_margin = int((~mask_margin).sum())
        candidates = candidates[mask_margin]
        log["removed_low_margin"] = removed_margin

        if restaurant_id in self.restaurants.index:
            thresholds = json.loads(
                self.restaurants.loc[restaurant_id, "discount_thresholds"]
            )
            invalid = [
                t for t in thresholds if t.get("min_order", 0) <= 0
            ]
            log["invalid_promotions"] = len(invalid)

        log["output"] = len(candidates)
        return candidates, log

    # ────────────────────────────────────────────────────────
    # Filter B — Dietary Toggle
    # ────────────────────────────────────────────────────────
    def filter_b_dietary(
        self,
        candidates: pd.DataFrame,
        dietary_toggle: str,
        cart_items: pd.DataFrame,
        user_id: str,
        session_start: str | datetime,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply dietary filtering based on session toggle, veg days, and
        cart composition inference."""
        log: dict[str, Any] = {"input": len(candidates), "toggle": dietary_toggle}

        effective_toggle = dietary_toggle

        if user_id in self.users.index:
            veg_days: list[int] = json.loads(self.users.loc[user_id, "veg_days"])
            if veg_days:
                ts = (
                    pd.to_datetime(session_start)
                    if not isinstance(session_start, datetime)
                    else session_start
                )
                if ts.weekday() in veg_days:
                    effective_toggle = "veg"
                    log["veg_day_override"] = True

        log["effective_toggle"] = effective_toggle

        if effective_toggle == "veg":
            candidates = candidates[candidates["veg_flag"] == True]  # noqa: E712
        elif effective_toggle == "vegan":
            # Synthetic data lacks granular vegan tagging; treat same as veg
            candidates = candidates[candidates["veg_flag"] == True]  # noqa: E712
        elif effective_toggle == "non-veg":
            pass  # no filter
        else:
            # "none" — infer from cart composition
            if len(cart_items) > 0:
                cart_has_nonveg = not cart_items["veg_flag"].all()
                if cart_has_nonveg:
                    pass  # show both
                    log["inferred"] = "cart_has_nonveg_show_both"
                else:
                    # All-veg cart, no toggle set: default to veg but NO hard filter
                    log["inferred"] = "all_veg_cart_soft_preference"

        log["output"] = len(candidates)
        return candidates, log

    # ────────────────────────────────────────────────────────
    # Filter C — Cuisine Coherence
    # ────────────────────────────────────────────────────────
    def filter_c_cuisine_coherence(
        self,
        candidates: pd.DataFrame,
        cart_items: pd.DataFrame,
        restaurant_id: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Keep only candidates matching the dominant cuisine in the cart.
        Suppress subcategories already covered by a combo/thali."""
        log: dict[str, Any] = {"input": len(candidates)}

        if len(cart_items) == 0:
            dominant = self.restaurants.loc[restaurant_id, "primary_cuisine"]
            log["source"] = "restaurant_default"
        else:
            cuisine_counts = cart_items["cuisine_tag"].value_counts()
            dominant = cuisine_counts.idxmax()
            log["source"] = "cart_mode"

        log["dominant_cuisine"] = dominant
        candidates = candidates[candidates["cuisine_tag"] == dominant]
        log["after_cuisine_match"] = len(candidates)

        combo_suppressed: set[str] = set()
        if len(cart_items) > 0:
            combos_in_cart = cart_items[
                (cart_items["is_combo"] == True)  # noqa: E712
                & (cart_items["subcategory"].isin(["thali", "meal_combo"]))
            ]
            for _, combo_row in combos_in_cart.iterrows():
                components = json.loads(combo_row["combo_components"])
                combo_suppressed.update(components)

        if combo_suppressed:
            candidates = candidates[
                ~candidates["subcategory"].isin(combo_suppressed)
            ]
            log["combo_suppressed_subcats"] = sorted(combo_suppressed)

        log["output"] = len(candidates)
        return candidates, log

    # ────────────────────────────────────────────────────────
    # Filter D — Quantity Saturation
    # ────────────────────────────────────────────────────────
    def filter_d_quantity_saturation(
        self,
        candidates: pd.DataFrame,
        cart_items: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Exclude candidates whose subcategory already meets or exceeds
        the configurable cap in the current cart."""
        log: dict[str, Any] = {"input": len(candidates)}

        if len(cart_items) == 0:
            log["output"] = len(candidates)
            return candidates, log

        cart_subcat_counts: Counter[str] = Counter(
            cart_items["subcategory"].tolist()
        )
        cart_size = len(cart_items)

        scale = 1.0
        if cart_size > GROUP_ORDER_SIZE:
            scale = 1.0 + (cart_size - GROUP_ORDER_SIZE) * 0.2
        log["scale_factor"] = round(scale, 2)

        def _is_saturated(subcat: str) -> bool:
            cap = self.saturation_caps.get(subcat, DEFAULT_CAP)
            effective_cap = math.ceil(cap * scale)
            return cart_subcat_counts.get(subcat, 0) >= effective_cap

        saturated_mask = candidates["subcategory"].apply(_is_saturated)
        log["removed_saturated"] = int(saturated_mask.sum())
        candidates = candidates[~saturated_mask]
        log["output"] = len(candidates)
        return candidates, log

    # ────────────────────────────────────────────────────────
    # Filter E — Deduplication and Recommendation Fatigue
    # ────────────────────────────────────────────────────────
    def filter_e_dedup_fatigue(
        self,
        candidates: pd.DataFrame,
        cart_item_ids: set[str],
        ignore_tracker: dict[str, int] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Remove items already in cart and suppress fatigued categories."""
        log: dict[str, Any] = {"input": len(candidates)}
        ignore_tracker = ignore_tracker or {}

        dedup_mask = candidates["item_id"].isin(cart_item_ids)
        log["removed_duplicates"] = int(dedup_mask.sum())
        candidates = candidates[~dedup_mask]

        suppressed_cats = {
            cat
            for cat, count in ignore_tracker.items()
            if count >= FATIGUE_THRESHOLD
        }
        if suppressed_cats:
            fatigue_mask = candidates["category"].isin(suppressed_cats)
            log["removed_fatigue"] = int(fatigue_mask.sum())
            log["suppressed_categories"] = sorted(suppressed_cats)
            candidates = candidates[~fatigue_mask]
        else:
            log["removed_fatigue"] = 0

        log["output"] = len(candidates)
        return candidates, log

    # ────────────────────────────────────────────────────────
    # Pipeline Orchestrator
    # ────────────────────────────────────────────────────────
    def run_filters(
        self,
        restaurant_id: str,
        user_id: str,
        session_start: str | datetime,
        dietary_toggle: str,
        cart_item_ids: set[str] | list[str],
        ignore_tracker: dict[str, int] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run all five filters in order A → B → C → D → E.

        Parameters
        ----------
        restaurant_id : restaurant being ordered from
        user_id : current user
        session_start : ISO timestamp or datetime of session start
        dietary_toggle : one of "veg", "vegan", "non-veg", "none"
        cart_item_ids : set of item_id strings currently in the cart
        ignore_tracker : category → consecutive-ignore count for fatigue

        Returns
        -------
        (filtered_candidates, filter_log) where filter_log contains
        counts at every stage.
        """
        cart_item_ids = set(cart_item_ids)
        filter_log: dict[str, Any] = {}

        rest_menu = self.menu[
            self.menu["restaurant_id"] == restaurant_id
        ].copy()
        filter_log["initial"] = len(rest_menu)

        cart_items = rest_menu[rest_menu["item_id"].isin(cart_item_ids)]

        # A — Availability and Margin
        candidates, log_a = self.filter_a_availability_margin(
            rest_menu, restaurant_id
        )
        filter_log["after_A"] = len(candidates)
        filter_log["log_A"] = log_a

        # B — Dietary Toggle
        candidates, log_b = self.filter_b_dietary(
            candidates, dietary_toggle, cart_items, user_id, session_start
        )
        filter_log["after_B"] = len(candidates)
        filter_log["log_B"] = log_b

        # C — Cuisine Coherence
        candidates, log_c = self.filter_c_cuisine_coherence(
            candidates, cart_items, restaurant_id
        )
        filter_log["after_C"] = len(candidates)
        filter_log["log_C"] = log_c

        # D — Quantity Saturation
        candidates, log_d = self.filter_d_quantity_saturation(
            candidates, cart_items
        )
        filter_log["after_D"] = len(candidates)
        filter_log["log_D"] = log_d

        # E — Deduplication and Fatigue
        candidates, log_e = self.filter_e_dedup_fatigue(
            candidates, cart_item_ids, ignore_tracker
        )
        filter_log["after_E"] = len(candidates)
        filter_log["log_E"] = log_e

        return candidates, filter_log
