"""
CSAO Rail Recommendation System — MBA Baseline (Market Basket Analysis)

Industry-standard baseline using Apriori association rules.
Its failures map directly to every design decision in the main system:
  - Same rules for every user (no segment awareness)
  - Ignores cart stage, abandonment risk, position quality
  - Cold start failure for new restaurants with no history

Inputs:
    data/cart_events.csv
    data/sessions.csv
    data/menu_items.csv
    data/training_features.csv  (for aligned evaluation)

Output:
    Evaluation metrics dict for comparison with the main system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Transaction baskets
# ---------------------------------------------------------------------------


def build_transaction_baskets(
    cart_events: pd.DataFrame,
    sessions: pd.DataFrame,
) -> List[List[str]]:
    """Build item baskets from completed sessions (≥ 2 items)."""

    completed_sids = set(
        sessions.loc[sessions["order_completed"] == True, "session_id"]  # noqa: E712
    )
    ce = cart_events[cart_events["session_id"].isin(completed_sids)].copy()

    in_cart = ce[
        (ce["was_recommendation"] == False)  # noqa: E712
        | (ce["was_accepted"] == True)  # noqa: E712
    ]

    baskets: List[List[str]] = []
    for _, grp in in_cart.groupby("session_id"):
        items = grp["item_id"].unique().tolist()
        if len(items) >= 2:
            baskets.append([str(i) for i in items])

    return baskets


# ---------------------------------------------------------------------------
# Apriori + association rules
# ---------------------------------------------------------------------------


def train_mba(
    baskets: List[List[str]],
    min_support: float = 0.003,
    min_confidence: float = 0.05,
    metric: str = "lift",
    min_threshold: float = 1.0,
) -> pd.DataFrame:
    """Run Apriori and generate association rules."""

    te = TransactionEncoder()
    te_array = te.fit_transform(baskets)
    df = pd.DataFrame(te_array, columns=te.columns_)

    frequent = apriori(df, min_support=min_support, use_colnames=True)
    if len(frequent) == 0:
        print("  [MBA] No frequent itemsets found — try lowering min_support.")
        return pd.DataFrame()

    rules = association_rules(frequent, metric=metric, min_threshold=min_threshold)
    print(
        f"  [MBA] {len(baskets):,} baskets -> {len(frequent):,} frequent "
        f"itemsets -> {len(rules):,} rules"
    )
    return rules


# ---------------------------------------------------------------------------
# Recommendation function
# ---------------------------------------------------------------------------


def mba_recommend(
    cart_items: List[str],
    rules: pd.DataFrame,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """Given current cart items, return top-k MBA recommendations.

    Score = confidence × lift for the best matching rule per consequent item.
    """
    if rules is None or len(rules) == 0:
        return []

    cart_set = frozenset(cart_items)
    candidates: Dict[str, float] = {}

    for _, rule in rules.iterrows():
        if rule["antecedents"].issubset(cart_set):
            for item in rule["consequents"]:
                if item not in cart_set:
                    score = float(rule["confidence"] * rule["lift"])
                    if item not in candidates or score > candidates[item]:
                        candidates[item] = score

    sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
    return sorted_candidates[:top_k]


# ---------------------------------------------------------------------------
# Evaluation (aligned with the main system's test set)
# ---------------------------------------------------------------------------


def _ndcg_at_k(relevance: List[float], k: int) -> float:
    """NDCG@k from a relevance list (already in ranked order)."""
    rel = relevance[:k]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_mba(
    rules: pd.DataFrame,
    test_features: pd.DataFrame,
    cart_events: pd.DataFrame,
    k: int = 8,
) -> Dict[str, float]:
    """Evaluate MBA on the same test impressions used for the main model.

    For each test session, at each recommendation-display timestamp:
      1. Reconstruct current cart from prior events.
      2. Ask MBA for top-k recommendations.
      3. Check overlap with actually-accepted items.

    Returns precision@k, ndcg@k, and hit-rate.
    """
    if rules is None or len(rules) == 0:
        return {"precision_at_k": 0.0, "ndcg_at_k": 0.0, "hit_rate": 0.0, "n_eval": 0}

    ce = cart_events.copy()
    ce["timestamp"] = pd.to_datetime(ce["timestamp"])
    events_by_session = {
        sid: grp.sort_values("timestamp").reset_index(drop=True)
        for sid, grp in ce.groupby("session_id")
    }

    tf = test_features.copy()
    tf["event_timestamp"] = pd.to_datetime(tf["event_timestamp"])

    all_precisions: List[float] = []
    all_ndcgs: List[float] = []
    all_hits: List[int] = []

    for session_id, sess_feats in tf.groupby("session_id"):
        sess_events = events_by_session.get(session_id)
        if sess_events is None:
            continue

        # Deduplicate by unique timestamps within the session
        for ts, ts_group in sess_feats.groupby("event_timestamp"):
            past = sess_events[sess_events["timestamp"] < ts]
            in_cart = past[
                (past["was_recommendation"] == False)  # noqa: E712
                | (past["was_accepted"] == True)  # noqa: E712
            ]
            cart_items = [str(i) for i in in_cart["item_id"].tolist()]
            if not cart_items:
                continue

            recs = mba_recommend(cart_items, rules, top_k=k)
            rec_items = [r[0] for r in recs]

            accepted = set(
                ts_group.loc[ts_group["label_accept"] == 1, "item_id"].astype(str)
            )
            if not accepted:
                continue

            hits = len(set(rec_items) & accepted)
            denom = min(k, len(rec_items)) if rec_items else 1
            all_precisions.append(hits / denom)

            relevance = [1.0 if r in accepted else 0.0 for r in rec_items[:k]]
            all_ndcgs.append(_ndcg_at_k(relevance, k))
            all_hits.append(1 if hits > 0 else 0)

    return {
        "precision_at_k": float(np.mean(all_precisions)) if all_precisions else 0.0,
        "ndcg_at_k": float(np.mean(all_ndcgs)) if all_ndcgs else 0.0,
        "hit_rate": float(np.mean(all_hits)) if all_hits else 0.0,
        "n_eval": len(all_precisions),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> Tuple[pd.DataFrame, Dict[str, float]]:
    print("CSAO Rail — MBA Baseline")
    print("-" * 48)

    cart_events = pd.read_csv(DATA_DIR / "cart_events.csv")
    sessions = pd.read_csv(DATA_DIR / "sessions.csv")
    features = pd.read_csv(DATA_DIR / "training_features.csv")

    # Use only training-period sessions for building rules
    sessions["start_time"] = pd.to_datetime(sessions["start_time"])
    train_sessions = sessions[sessions["start_time"] < "2025-12-22"]
    test_sessions = sessions[sessions["start_time"] >= "2025-12-29"]

    print("Building transaction baskets (train period) ...")
    baskets = build_transaction_baskets(cart_events, train_sessions)
    print(f"  {len(baskets):,} baskets")

    print("Running Apriori ...")
    rules = train_mba(baskets)

    # Evaluate on test split
    features["event_timestamp"] = pd.to_datetime(features["event_timestamp"])
    sess_start = sessions.set_index("session_id")["start_time"]
    feat_sess_start = features["session_id"].map(sess_start)
    test_mask = feat_sess_start >= "2025-12-29"
    test_features = features[test_mask]

    print(f"\nEvaluating MBA on {len(test_features):,} test impressions ...")
    metrics = evaluate_mba(rules, test_features, cart_events, k=8)
    print(f"  Precision@8 = {metrics['precision_at_k']:.4f}")
    print(f"  NDCG@8      = {metrics['ndcg_at_k']:.4f}")
    print(f"  Hit rate     = {metrics['hit_rate']:.4f}")
    print(f"  Evaluated    = {metrics['n_eval']:,} impression groups")

    return rules, metrics


if __name__ == "__main__":
    main()
