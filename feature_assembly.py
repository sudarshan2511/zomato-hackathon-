"""
CSAO Rail Recommendation System — Step 4: Feature Assembly

Builds the static + dynamic feature matrix used by the downstream
GRU encoder (Step 5) and LightGBM ranker.

Inputs (from generate_data.py):
    data/restaurants.csv
    data/menu_items.csv
    data/users.csv
    data/sessions.csv
    data/cart_events.csv

Output:
    data/training_features.csv

Each row represents a single recommendation impression
(`cart_events.was_recommendation == True`) with:
    - static user, item, restaurant features
    - dynamic cart and session features at the moment of display
    - label / metadata columns for later model training
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "training_features.csv"


# ---------------------------------------------------------------------------
# Helper dataclasses for cart snapshot
# ---------------------------------------------------------------------------


@dataclass
class CartSnapshot:
    item_ids: List[str]
    categories: List[str]
    subcategories: List[str]
    prices: List[float]

    @property
    def size(self) -> int:
        return len(self.item_ids)

    @property
    def total_value(self) -> float:
        return float(sum(self.prices)) if self.prices else 0.0


# ---------------------------------------------------------------------------
# Static feature assembly
# ---------------------------------------------------------------------------


def build_static_feature_tables(
    restaurants: pd.DataFrame,
    menu: pd.DataFrame,
    users: pd.DataFrame,
    cart_events: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare static user/item/restaurant feature tables.

    Also computes a restaurant-level "add-on success rate" from
    the synthetic cart_events logs.
    """
    # Restaurant-level add-on success rate
    rec_events = cart_events[cart_events["was_recommendation"] == True]  # noqa: E712
    if len(rec_events) > 0:
        rest_addon = (
            rec_events.groupby("restaurant_id")["was_accepted"]
            .mean()
            .rename("restaurant_addon_success_rate")
            .reset_index()
        )
    else:
        rest_addon = pd.DataFrame(columns=["restaurant_id", "restaurant_addon_success_rate"])

    restaurants_static = restaurants.copy()
    restaurants_static = restaurants_static.merge(
        rest_addon, on="restaurant_id", how="left"
    )
    restaurants_static["restaurant_addon_success_rate"] = restaurants_static[
        "restaurant_addon_success_rate"
    ].fillna(0.0)

    # Item static table — thin projection (+ meal-time popularity expansion)
    item_cols = [
        "item_id",
        "restaurant_id",
        "price",
        "category",
        "subcategory",
        "cuisine_tag",
        "veg_flag",
        "bestseller_flag",
        "availability",
        "margin_pct",
        "prep_time_mins",
        "popularity_score",
        "popularity_by_meal",
    ]
    items_static = menu[item_cols].copy()

    # Expand popularity_by_meal JSON → separate numeric columns
    if "popularity_by_meal" in items_static.columns:
        def _expand_popularity(val: object) -> pd.Series:
            try:
                data = json.loads(val) if isinstance(val, str) else (val or {})
            except Exception:
                data = {}
            return pd.Series(
                {
                    "item_pop_breakfast": float(data.get("breakfast", 0.0)),
                    "item_pop_lunch": float(data.get("lunch", 0.0)),
                    "item_pop_dinner": float(data.get("dinner", 0.0)),
                    "item_pop_late_night": float(data.get("late_night", 0.0)),
                }
            )

        pop_expanded = items_static["popularity_by_meal"].apply(_expand_popularity)
        items_static = pd.concat(
            [items_static.drop(columns=["popularity_by_meal"]), pop_expanded],
            axis=1,
        )

    # User static table — keep main segmentation + RFM metrics
    user_cols = [
        "user_id",
        "segment",
        "city",
        "dietary_preference",
        "veg_days",
        "rfm_recency",
        "rfm_frequency",
        "rfm_monetary",
        "order_count",
        "avg_order_value",
    ]
    users_static = users[user_cols].copy()

    return restaurants_static, items_static, users_static


# ---------------------------------------------------------------------------
# Dynamic feature helpers
# ---------------------------------------------------------------------------


MAIN_CATS = {"main", "combo"}
CARB_SUBCATS = {
    "rice",
    "biryani",
    "naan",
    "roti",
    "paratha",
    "pav_bhaji",
    "bhature",
    "pizza",
    "pasta",
    "bread",
}
BEV_CATS = {"beverage"}
BEV_SUBCATS = {"soft_drink", "lassi", "coffee", "tea", "juice", "mocktail", "soda"}
DESSERT_CATS = {"dessert"}
DESSERT_SUBCATS = {
    "indian_sweet",
    "western_dessert",
    "cake",
    "brownie",
    "mousse",
    "falooda",
    "ice_cream",
}
BREAD_SUBCATS = {"naan", "roti", "paratha", "bread"}


def _compute_cart_stage(snapshot: CartSnapshot, first_item_price: float) -> int:
    """Map cart composition to a discrete stage [0–3]."""
    if snapshot.size == 0:
        return 0

    cats = set(snapshot.categories)
    subcats = set(snapshot.subcategories)

    has_main = bool(cats & MAIN_CATS)
    has_carb = bool(subcats & CARB_SUBCATS)
    has_bev = bool(cats & BEV_CATS or subcats & BEV_SUBCATS)
    has_dessert = bool(cats & DESSERT_CATS or subcats & DESSERT_SUBCATS)

    if not has_main:
        # Some carts will be odd (only sides / dessert etc.)
        return 1

    if has_main and not has_carb:
        return 1

    if has_main and has_carb and not (has_bev or has_dessert):
        return 2

    # Main + carb + at least one "finishing" item
    return 3


def _gap_flags(snapshot: CartSnapshot, candidate_row: pd.Series) -> Dict[str, int]:
    """Detect which meal components are missing and whether candidate fills them."""
    cats = set(snapshot.categories)
    subcats = set(snapshot.subcategories)

    has_bev = bool({"beverage"} & cats or BEV_SUBCATS & subcats)
    has_dessert = bool({"dessert"} & cats or DESSERT_SUBCATS & subcats)
    has_bread = bool(BREAD_SUBCATS & subcats)

    cand_cat = candidate_row["category"]
    cand_sub = candidate_row["subcategory"]

    fills_bev = int(
        (not has_bev)
        and (
            cand_cat == "beverage"
            or cand_sub in BEV_SUBCATS
        )
    )
    fills_dessert = int(
        (not has_dessert)
        and (
            cand_cat == "dessert"
            or cand_sub in DESSERT_SUBCATS
        )
    )
    fills_bread = int(
        (not has_bread)
        and (cand_sub in BREAD_SUBCATS)
    )

    return {
        "gap_missing_beverage": int(not has_bev),
        "gap_missing_dessert": int(not has_dessert),
        "gap_missing_bread": int(not has_bread),
        "candidate_fills_beverage_gap": fills_bev,
        "candidate_fills_dessert_gap": fills_dessert,
        "candidate_fills_bread_gap": fills_bread,
    }


def _complement_score(snapshot: CartSnapshot, candidate_row: pd.Series) -> float:
    """Very lightweight heuristic complement score.

    - Biryani in cart boosts salan / raita / beverage sides
    - Main course boosts bread and beverage
    """
    subcats = set(snapshot.subcategories)
    cats = set(snapshot.categories)

    cand_cat = candidate_row["category"]
    cand_sub = candidate_row["subcategory"]

    score = 0.0

    if "biryani" in subcats:
        if cand_sub in {"salan", "accompaniment"}:
            score += 1.0
        if cand_cat == "beverage":
            score += 0.5

    if "main" in cats or "combo" in cats:
        if cand_sub in BREAD_SUBCATS:
            score += 0.8
        if cand_cat == "beverage":
            score += 0.5

    return float(score)


def _price_anchor_features(
    snapshot: CartSnapshot, first_item_price: float, candidate_price: float
) -> Dict[str, float]:
    if first_item_price <= 0:
        return {
            "price_anchor_ratio": 1.0,
            "price_anchor_diff": float(candidate_price),
        }
    ratio = float(candidate_price) / float(first_item_price)
    diff = float(candidate_price) - float(first_item_price)
    return {
        "price_anchor_ratio": ratio,
        "price_anchor_diff": diff,
    }


def _distance_to_discount_features(
    restaurant_row: pd.Series, snapshot: CartSnapshot, candidate_price: float
) -> Dict[str, float]:
    """Compute distance-to-discount style features from thresholds JSON.

    Includes:
        - dtd_gap: rupee gap to nearest discount threshold
        - dtd_closes_gap: candidate crosses some threshold
        - dtd_overshoot: candidate overshoots gap by > 1.8x
        - dtd_free_delivery_unlock: specifically unlocks free delivery
        - dtd_nudge_urgency: heuristic 0–1 score of "good nudge"
    """
    thresholds_raw = restaurant_row.get("discount_thresholds", "[]")
    try:
        thresholds = json.loads(thresholds_raw)
    except Exception:
        thresholds = []

    cart_value = snapshot.total_value
    best_gap = None
    closes_gap_flag = 0
    overshoot_flag = 0
    free_delivery_unlock = 0

    for t in thresholds:
        min_order = float(t.get("min_order", 0))
        if min_order <= 0:
            continue
        gap = max(0.0, min_order - cart_value)
        if best_gap is None or gap < best_gap:
            best_gap = gap

    if best_gap is None:
        return {
            "dtd_gap": 0.0,
            "dtd_closes_gap": 0.0,
            "dtd_overshoot": 0.0,
            "dtd_free_delivery_unlock": 0.0,
            "dtd_nudge_urgency": 0.0,
        }

    if best_gap <= candidate_price + 1e-6:
        closes_gap_flag = 1.0
    if candidate_price > 0 and candidate_price > 1.8 * best_gap:
        overshoot_flag = 1.0

    # Free-delivery unlock flag: uses restaurant-level free_delivery_min
    free_del_min = float(restaurant_row.get("free_delivery_min", 0) or 0)
    if free_del_min > 0 and cart_value < free_del_min <= cart_value + candidate_price + 1e-6:
        free_delivery_unlock = 1.0

    # Heuristic nudge urgency score in [0, 1]
    # High when gap is small, candidate closes gap, and overshoot is low.
    max_considered_gap = 250.0
    norm_gap = min(best_gap, max_considered_gap) / max_considered_gap
    base = 1.0 - norm_gap  # smaller gap → closer to 1
    if not closes_gap_flag:
        base *= 0.2
    if overshoot_flag:
        base *= 0.1
    nudge_urgency = float(max(0.0, min(1.0, base)))

    return {
        "dtd_gap": float(best_gap),
        "dtd_closes_gap": closes_gap_flag,
        "dtd_overshoot": overshoot_flag,
        "dtd_free_delivery_unlock": free_delivery_unlock,
        "dtd_nudge_urgency": nudge_urgency,
    }


def _peak_hour_mode(ts: datetime) -> str:
    h = ts.hour
    if 12 <= h < 14:
        return "lunch_C2O"
    if 19 <= h < 22:
        return "dinner_AOV"
    if 23 <= h or h < 5:
        return "late_night_impulse"
    return "normal"


def _seasonal_weight_bucket(ts: datetime) -> str:
    m = ts.month
    if m in {3, 4, 5, 6}:  # warmer months
        return "summer_bev_dessert_up"
    if m in {11, 12, 1, 2}:  # cooler months
        return "winter_warm_dessert_up"
    return "neutral"


def _abandonment_risk_score(
    snapshot: CartSnapshot,
    time_since_last_add: float,
    consecutive_rejections: int,
) -> float:
    """Heuristic 0–1 score estimating current abandonment risk.

    Higher when:
      - cart is very small,
      - user has been idle for a while,
      - user has rejected many consecutive recommendations.
    """
    # Cart size: very small carts are more fragile
    if snapshot.size == 0:
        size_term = 1.0
    elif snapshot.size == 1:
        size_term = 0.8
    elif snapshot.size <= 3:
        size_term = 0.4
    else:
        size_term = 0.2

    # Idle time (cap at 5 minutes)
    idle_norm = min(time_since_last_add, 300.0) / 300.0  # 0–1

    # Rejection streak (cap at 5)
    rej_norm = min(consecutive_rejections, 5) / 5.0  # 0–1

    # Weighted combination, then clamp to [0, 1]
    raw = 0.4 * size_term + 0.3 * idle_norm + 0.3 * rej_norm
    return float(max(0.0, min(1.0, raw)))


# ---------------------------------------------------------------------------
# Main feature assembly
# ---------------------------------------------------------------------------


def assemble_features(
    restaurants_csv: str | Path = DATA_DIR / "restaurants.csv",
    menu_csv: str | Path = DATA_DIR / "menu_items.csv",
    users_csv: str | Path = DATA_DIR / "users.csv",
    sessions_csv: str | Path = DATA_DIR / "sessions.csv",
    cart_events_csv: str | Path = DATA_DIR / "cart_events.csv",
    output_path: str | Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """End-to-end feature assembly for Step 4.

    Returns the assembled feature DataFrame and writes it to ``output_path``.
    """
    restaurants = pd.read_csv(restaurants_csv)
    menu = pd.read_csv(menu_csv)
    users = pd.read_csv(users_csv)
    sessions = pd.read_csv(sessions_csv)
    cart_events = pd.read_csv(cart_events_csv)

    # Attach restaurant_id to cart_events via sessions (needed for rest-level stats)
    if "restaurant_id" not in cart_events.columns:
        cart_events = cart_events.merge(
            sessions[["session_id", "restaurant_id"]],
            on="session_id",
            how="left",
        )

    # Build static tables
    rest_static, item_static, user_static = build_static_feature_tables(
        restaurants, menu, users, cart_events
    )

    rest_static_idx = rest_static.set_index("restaurant_id")
    item_static_idx = item_static.set_index("item_id")
    user_static_idx = user_static.set_index("user_id")
    menu_idx = menu.set_index("item_id")

    # Pre-index events and sessions
    cart_events["timestamp"] = pd.to_datetime(cart_events["timestamp"])
    sessions_idx = sessions.set_index("session_id")

    # Consider only recommendation impressions for training
    rec_events = cart_events[cart_events["was_recommendation"] == True].copy()  # noqa: E712
    rec_events = rec_events.sort_values(["session_id", "timestamp"]).reset_index(drop=True)

    # For cart reconstruction we need per-session ordered events
    events_by_session: Dict[str, pd.DataFrame] = {
        sid: grp.sort_values("timestamp").reset_index(drop=True)
        for sid, grp in cart_events.groupby("session_id")
    }

    feature_rows: List[Dict[str, object]] = []

    # Iterate over each recommendation impression
    for _, evt in rec_events.iterrows():
        session_id = evt["session_id"]
        item_id = evt["item_id"]
        ts: datetime = evt["timestamp"]

        sess_row = sessions_idx.loc[session_id]
        user_id = sess_row["user_id"]
        restaurant_id = sess_row["restaurant_id"]

        # Static slices
        user_row = user_static_idx.loc[user_id]
        item_row = item_static_idx.loc[item_id]
        rest_row = rest_static_idx.loc[restaurant_id]

        # Reconstruct cart snapshot BEFORE this event
        evts_sess = events_by_session[session_id]
        past_mask = evts_sess["timestamp"] < ts
        past = evts_sess[past_mask]

        in_cart = past[
            (past["was_recommendation"] == False)  # organic adds
            | (past["was_accepted"] == True)  # accepted recs
        ]

        if len(in_cart) == 0:
            snapshot = CartSnapshot([], [], [], [])
            first_price = float(item_row["price"])
        else:
            item_ids_in_cart = in_cart["item_id"].tolist()
            menu_rows = menu_idx.loc[item_ids_in_cart]
            snapshot = CartSnapshot(
                item_ids=item_ids_in_cart,
                categories=menu_rows["category"].tolist(),
                subcategories=menu_rows["subcategory"].tolist(),
                prices=menu_rows["price"].astype(float).tolist(),
            )
            # First cart item is simply the first organic or accepted event
            first_evt = in_cart.iloc[0]
            first_item_price = float(menu_idx.loc[first_evt["item_id"], "price"])
            first_price = first_item_price

        # Dynamic cart features
        cart_stage = _compute_cart_stage(snapshot, first_price)
        gap_feat = _gap_flags(snapshot, item_row)
        comp_score = _complement_score(snapshot, item_row)
        price_anchor = _price_anchor_features(snapshot, first_price, float(item_row["price"]))
        dtd_feat = _distance_to_discount_features(rest_row, snapshot, float(item_row["price"]))

        # Session temporal signals
        time_since_last_add = 0.0
        consecutive_rejections = 0

        if len(past) > 0:
            # time since last cart-affecting event (organic or accepted)
            cart_affecting = past[
                (past["was_recommendation"] == False)
                | (past["was_accepted"] == True)
            ]
            if len(cart_affecting) > 0:
                last_ts = cart_affecting["timestamp"].max()
                time_since_last_add = (ts - last_ts).total_seconds()

            # consecutive rejected recommendations right before this impression
            rev = past.iloc[::-1]
            for _, r in rev.iterrows():
                if r["was_recommendation"] != True:  # noqa: E712
                    break
                if r["was_accepted"] == True:  # noqa: E712
                    break
                consecutive_rejections += 1

        # Peak hour / seasonal
        peak_mode = _peak_hour_mode(ts)
        season_bucket = _seasonal_weight_bucket(ts)

        # Profile veg days signal (hard veg day equivalent to toggle)
        veg_days = json.loads(user_row["veg_days"])
        profile_veg_today = int(ts.weekday() in veg_days)

        # Abandonment risk composite score
        abandon_risk = _abandonment_risk_score(
            snapshot=snapshot,
            time_since_last_add=time_since_last_add,
            consecutive_rejections=consecutive_rejections,
        )

        # Label / metadata from cart_events & sessions
        row: Dict[str, object] = {
            # IDs
            "session_id": session_id,
            "user_id": user_id,
            "restaurant_id": restaurant_id,
            "item_id": item_id,
            # Labels / training metadata
            "label_accept": int(bool(evt["was_accepted"])),
            "label_position": int(evt.get("position_shown", 0) or 0),
            "label_cart_abandoned": int(not bool(sess_row["order_completed"])),
            "final_order_value": float(sess_row["final_order_value"]),
            # Simple price-derived proxy for AOV contribution label
            "label_aov_proxy": float(item_row["price"]),
            "event_timestamp": ts.isoformat(),
        }

        # Static user features
        for col in user_static.columns:
            if col == "user_id":
                continue
            row[f"user_{col}"] = user_row[col]

        # Static item features
        for col in item_static.columns:
            if col in ("item_id", "restaurant_id"):
                continue
            row[f"item_{col}"] = item_row[col]

        # Static restaurant features
        for col in rest_static.columns:
            if col == "restaurant_id":
                continue
            row[f"rest_{col}"] = rest_row[col]

        # Dynamic features
        row.update(
            {
                "cart_size": snapshot.size,
                "cart_total_value": snapshot.total_value,
                "cart_stage": cart_stage,
                "time_since_last_add_sec": time_since_last_add,
                "consecutive_rejections": consecutive_rejections,
                "complement_score": comp_score,
                "peak_hour_mode": peak_mode,
                "season_bucket": season_bucket,
                "profile_veg_day_flag": profile_veg_today,
                "abandonment_risk_score": abandon_risk,
            }
        )
        row.update(gap_feat)
        row.update(price_anchor)
        row.update(dtd_feat)

        feature_rows.append(row)

    features_df = pd.DataFrame(feature_rows)

    os.makedirs(os.path.dirname(str(output_path)) or ".", exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"[Step 4] Assembled {len(features_df)} feature rows -> {output_path}")
    return features_df


def main() -> None:
    print("CSAO Rail — Step 4: Feature Assembly")
    print("-" * 48)
    assemble_features()


if __name__ == "__main__":
    main()

