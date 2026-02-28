"""
CSAO Rail Recommendation System — Step 6: Post-Ranking Processing

Six deterministic business-rule layers applied sequentially after
LightGBM scoring to produce the final 8–10 item CSAO rail.

Layers:
    1. Subcategory Diversity   — max 2 items per subcategory
    2. Category Mix            — ensure side / beverage / dessert coverage
    3. Price Shock Check       — within ±30 % of cart average
    4. Margin Cap              — ≤ 30 % of rail may be ultra-high-margin
    5. Time-of-Day Exclusions  — no breakfast at dinner, no heavy mains at breakfast
    6. Distance-to-Discount    — gap-closing item forced to position 1

Input:
    A DataFrame of scored candidates (one row per candidate) sorted by
    business_score descending.  Must contain at minimum the columns
    listed in _REQUIRED_COLS.

Output:
    A trimmed, re-ordered DataFrame of 8–10 items ready for UI display
    with rank positions and an explanation for position 1.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

MAX_PER_SUBCATEGORY = 2
DESIRED_MIX_CATS = {"side", "beverage", "dessert"}
PRICE_SHOCK_TOLERANCE = 0.30            # ±30 %
ULTRA_HIGH_MARGIN_THRESHOLD = 55.0      # margin_pct above this is "ultra-high"
MARGIN_CAP_FRACTION = 0.30              # max share of rail that can be ultra-high
DTD_URGENCY_THRESHOLD = 0.70            # nudge urgency must exceed this

BREAKFAST_ONLY_SUBCATS = {
    "idli", "vada", "dosa", "uttapam", "pongal",
    "croissant", "waffle", "muffin",
}
HEAVY_MAIN_SUBCATS = {
    "biryani", "steak", "pizza", "pasta", "burger",
    "curry", "manchurian", "noodles", "fried_rice",
}

_REQUIRED_COLS = {
    "item_id", "item_category", "item_subcategory", "item_price",
    "item_margin_pct", "business_score",
}


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 1 — SUBCATEGORY DIVERSITY
# ═══════════════════════════════════════════════════════════════════════════


def filter_subcategory_diversity(
    df: pd.DataFrame,
    max_per_sub: int = MAX_PER_SUBCATEGORY,
) -> pd.DataFrame:
    """Keep at most *max_per_sub* items per subcategory (top by score)."""
    return (
        df
        .sort_values("business_score", ascending=False)
        .groupby("item_subcategory", sort=False)
        .head(max_per_sub)
        .sort_values("business_score", ascending=False)
        .reset_index(drop=True)
    )


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2 — CATEGORY MIX ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════


def enforce_category_mix(
    rail: pd.DataFrame,
    full_pool: pd.DataFrame,
    desired_cats: set = DESIRED_MIX_CATS,
    rail_size: int = 10,
) -> pd.DataFrame:
    """Pull in missing desired categories from *full_pool* if absent.

    Replaces the lowest-scored rail item with the highest-scored
    candidate of each missing category.
    """
    rail = rail.copy().sort_values("business_score", ascending=False).reset_index(drop=True)
    present_cats = set(rail["item_category"].unique())

    for cat in desired_cats:
        if cat in present_cats:
            continue
        pool_cat = full_pool[
            (full_pool["item_category"] == cat)
            & (~full_pool["item_id"].isin(rail["item_id"]))
        ]
        if pool_cat.empty:
            continue
        best = pool_cat.sort_values("business_score", ascending=False).iloc[0]
        if len(rail) >= rail_size:
            rail = rail.iloc[:-1]
        best_df = pd.DataFrame([best])
        rail = pd.concat([rail, best_df], ignore_index=True)
        present_cats.add(cat)

    return rail.sort_values("business_score", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 3 — PRICE SHOCK CHECK
# ═══════════════════════════════════════════════════════════════════════════


def filter_price_shock(
    df: pd.DataFrame,
    cart_total: float,
    cart_size: int,
    tolerance: float = PRICE_SHOCK_TOLERANCE,
) -> pd.DataFrame:
    """Remove candidates whose price deviates > *tolerance* from cart avg."""
    if cart_size <= 0:
        return df

    cart_avg = cart_total / cart_size
    lo = cart_avg * (1.0 - tolerance)
    hi = cart_avg * (1.0 + tolerance)
    return df[(df["item_price"] >= lo) & (df["item_price"] <= hi)].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 4 — MARGIN CAP
# ═══════════════════════════════════════════════════════════════════════════


def apply_margin_cap(
    df: pd.DataFrame,
    margin_threshold: float = ULTRA_HIGH_MARGIN_THRESHOLD,
    cap_fraction: float = MARGIN_CAP_FRACTION,
) -> pd.DataFrame:
    """Ensure no more than *cap_fraction* of the rail is ultra-high-margin."""
    df = df.sort_values("business_score", ascending=False).reset_index(drop=True)
    is_uhm = df["item_margin_pct"] > margin_threshold
    max_uhm = max(1, int(np.ceil(len(df) * cap_fraction)))

    if is_uhm.sum() <= max_uhm:
        return df

    uhm_indices = df.index[is_uhm].tolist()
    drop_count = int(is_uhm.sum()) - max_uhm
    to_drop = uhm_indices[-drop_count:]
    return df.drop(index=to_drop).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 5 — TIME-OF-DAY EXCLUSIONS
# ═══════════════════════════════════════════════════════════════════════════


def filter_time_of_day(
    df: pd.DataFrame,
    peak_mode: str,
) -> pd.DataFrame:
    """Exclude items inappropriate for the current time slot."""
    if peak_mode == "late_night_impulse":
        return df

    if peak_mode in ("lunch_C2O", "dinner_AOV"):
        mask = ~df["item_subcategory"].isin(BREAKFAST_ONLY_SUBCATS)
        return df[mask].reset_index(drop=True)

    if peak_mode == "normal":
        mask = ~df["item_subcategory"].isin(HEAVY_MAIN_SUBCATS)
        return df[mask].reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 6 — DISTANCE-TO-DISCOUNT OVERRIDE
# ═══════════════════════════════════════════════════════════════════════════


def apply_dtd_override(
    df: pd.DataFrame,
    urgency_threshold: float = DTD_URGENCY_THRESHOLD,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Force the best gap-closing item to position 1 if conditions are met.

    Returns (reordered_df, explanation_text_or_None).
    """
    if "dtd_nudge_urgency" not in df.columns:
        return df, None

    qual = df[
        (df["dtd_nudge_urgency"] > urgency_threshold)
        & (df["dtd_closes_gap"] == 1)
        & (df["dtd_overshoot"] == 0)
    ]

    if qual.empty:
        return df, None

    best_idx = qual["dtd_nudge_urgency"].idxmax()
    best_row = df.loc[best_idx]
    explanation = _build_dtd_explanation(best_row)

    rest = df.drop(index=best_idx)
    reordered = pd.concat(
        [df.loc[[best_idx]], rest], ignore_index=True
    )
    return reordered, explanation


def _build_dtd_explanation(row: pd.Series) -> str:
    """Human-readable nudge message for the D-t-D override item."""
    if row.get("dtd_free_delivery_unlock", 0):
        return f"Add {row.get('item_name', 'this')} to unlock free delivery!"
    gap = row.get("dtd_gap", 0)
    if gap > 0:
        return f"Just \u20b9{int(gap)} away from a discount \u2014 {row.get('item_name', 'this item')} gets you there!"
    return f"{row.get('item_name', 'This item')} completes your discount threshold!"


# ═══════════════════════════════════════════════════════════════════════════
# POSITION-1 EXPLANATION (non-D-t-D)
# ═══════════════════════════════════════════════════════════════════════════


def generate_explanation(row: pd.Series) -> str:
    """One-line explanation for the position-1 item based on its top signal."""
    if row.get("candidate_fills_beverage_gap", 0):
        return "Complete your meal with a drink!"
    if row.get("candidate_fills_dessert_gap", 0):
        return "End on a sweet note \u2014 add a dessert!"
    if row.get("candidate_fills_bread_gap", 0):
        return "Don\u2019t forget the bread!"
    if row.get("complement_score", 0) >= 1.0:
        return f"Perfect pairing with your cart!"
    if row.get("item_bestseller_flag", 0):
        return "Bestseller \u2014 most ordered with similar meals!"
    return "Recommended for you!"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


def apply_post_ranking(
    candidates: pd.DataFrame,
    cart_total: float,
    cart_size: int,
    peak_mode: str,
    rail_size: int = 10,
    *,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply all six post-ranking layers and return the final rail.

    Parameters
    ----------
    candidates : DataFrame
        Scored candidates sorted by ``business_score`` descending.
    cart_total : float
        Current cart monetary total.
    cart_size : int
        Number of items currently in the cart.
    peak_mode : str
        One of ``"normal"``, ``"lunch_C2O"``, ``"dinner_AOV"``,
        ``"late_night_impulse"``.
    rail_size : int
        Target number of items in the output rail (8–10).
    verbose : bool
        If True, print counts after each layer.

    Returns
    -------
    (rail_df, stats)
        ``rail_df`` has columns: rank, item_id, item_name, item_price,
        item_category, item_subcategory, business_score, explanation.
        ``stats`` records how many items each layer removed.
    """
    df = candidates.sort_values("business_score", ascending=False).reset_index(drop=True)
    full_pool = df.copy()
    stats: Dict[str, int] = {"input": len(df)}

    def _log(layer: str, after: pd.DataFrame) -> None:
        removed = stats.get("_prev", stats["input"]) - len(after)
        stats[layer] = removed
        stats["_prev"] = len(after)
        if verbose:
            print(f"  [{layer:25s}] removed {removed:>3d}  |  remaining {len(after):>4d}")

    # Layer 1 — Subcategory diversity
    df = filter_subcategory_diversity(df)
    _log("subcategory_diversity", df)

    # Layer 2 — Category mix (operates on top rail_size, pulls from full_pool)
    rail_top = df.head(rail_size).copy()
    rail_top = enforce_category_mix(rail_top, full_pool, rail_size=rail_size)
    df = pd.concat([rail_top, df[~df["item_id"].isin(rail_top["item_id"])]], ignore_index=True)
    _log("category_mix", df)

    # Layer 3 — Price shock
    df = filter_price_shock(df, cart_total, cart_size)
    _log("price_shock", df)

    # Layer 4 — Margin cap (on top rail_size slice)
    rail_top = df.head(rail_size).copy()
    rail_top = apply_margin_cap(rail_top)
    df = pd.concat([rail_top, df.iloc[len(rail_top):]], ignore_index=True)
    _log("margin_cap", df)

    # Layer 5 — Time-of-day exclusions
    df = filter_time_of_day(df, peak_mode)
    _log("time_of_day", df)

    # Trim to rail_size before D-t-D override
    df = df.head(rail_size).reset_index(drop=True)

    # Layer 6 — Distance-to-discount override
    df, dtd_explanation = apply_dtd_override(df)
    stats["dtd_override"] = 1 if dtd_explanation else 0

    # Assign ranks and explanations
    df = df.reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    explanations = [""] * len(df)
    if dtd_explanation:
        explanations[0] = dtd_explanation
    else:
        explanations[0] = generate_explanation(df.iloc[0])
    df["explanation"] = explanations

    stats.pop("_prev", None)
    stats["output"] = len(df)

    output_cols = [
        "rank", "item_id", "business_score",
        "item_price", "item_category", "item_subcategory", "explanation",
    ]
    if "item_name" in df.columns:
        output_cols.insert(2, "item_name")

    extra = [c for c in df.columns if c not in output_cols]
    final = df[output_cols + extra]

    return final, stats


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: score + post-rank in one call
# ═══════════════════════════════════════════════════════════════════════════


def score_and_postrank(
    models: dict,
    feature_row_df: pd.DataFrame,
    feature_cols: List[str],
    cart_total: float,
    cart_size: int,
    peak_mode: str,
    rail_size: int = 10,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Run LightGBM scoring then post-ranking in one call.

    ``feature_row_df`` must contain both the feature columns *and*
    the metadata columns (item_id, item_category, etc.).
    """
    from lgbm_ranker import compute_business_score

    X = feature_row_df[feature_cols]
    preds = {name: model.predict(X) for name, model in models.items()}
    scores = compute_business_score(preds)
    scored = feature_row_df.copy()
    scored["business_score"] = scores

    return apply_post_ranking(
        scored,
        cart_total=cart_total,
        cart_size=cart_size,
        peak_mode=peak_mode,
        rail_size=rail_size,
        verbose=verbose,
    )
