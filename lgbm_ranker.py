"""
CSAO Rail Recommendation System — Step 5b: LightGBM Multi-Objective Ranker

Five LightGBM models, each optimising one business objective:
    Accept  — LambdaRank   → P(user adds item)
    AOV     — Regression   → expected value contribution
    Abandon — Binary       → P(cart abandoned after showing this)
    Timing  — Binary       → P(fits current cart stage)
    Anchor  — Binary       → P(strong position-1 item)

Final business score:
    Score = 0.30·p_accept + 0.30·e_aov − 0.20·p_abandon
          + 0.10·p_timing + 0.10·p_anchor

Inputs:
    data/training_features.csv
    data/gru_hidden_states.npy      (from gru_encoder.py)
    data/sessions.csv

Outputs:
    models/lgbm_*.txt               (5 model files)
    Evaluation metrics and SHAP plots
"""

from __future__ import annotations

import os
import warnings
from itertools import product as iterproduct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


# ═══════════════════════════════════════════════════════════════════════════
# LABEL ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

# Stage → item categories considered "appropriate" at that stage
_STAGE_APPROPRIATE_CATS: Dict[int, set] = {
    0: {"main", "combo"},
    1: {"side", "appetizer", "bread", "rice"},
    2: {"beverage", "dessert", "side"},
    3: {"beverage", "dessert"},
}

_STAGE_APPROPRIATE_SUBCATS: Dict[int, set] = {
    1: {"naan", "roti", "paratha", "biryani", "fried_rice", "pulao", "rice",
        "salan", "raita", "accompaniment"},
    2: {"soft_drink", "lassi", "coffee", "tea", "juice", "mocktail", "soda",
        "indian_sweet", "western_dessert", "cake", "brownie", "mousse",
        "falooda", "ice_cream"},
    3: {"soft_drink", "lassi", "coffee", "tea", "juice", "mocktail", "soda",
        "indian_sweet", "western_dessert", "cake", "brownie", "mousse",
        "falooda", "ice_cream"},
}


def engineer_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all five label targets from raw columns."""
    out = df.copy()

    # Accept — already present as label_accept (0/1)

    # AOV — item price × accepted / 500  (continuous in [0, ~1])
    out["label_aov"] = out["label_aov_proxy"] * out["label_accept"] / 500.0

    # Abandon — already present as label_cart_abandoned (0/1)

    # Timing — accepted at an appropriate cart stage (vectorised)
    appropriate = pd.Series(False, index=out.index)
    for stage, ok_cats in _STAGE_APPROPRIATE_CATS.items():
        stage_mask = out["cart_stage"] == stage
        cat_ok = out["item_category"].isin(ok_cats)
        sub_ok = out["item_subcategory"].isin(
            _STAGE_APPROPRIATE_SUBCATS.get(stage, set())
        )
        appropriate = appropriate | (stage_mask & (cat_ok | sub_ok))

    out["label_timing"] = ((out["label_accept"] == 1) & appropriate).astype(int)

    # Anchor — accepted when shown at position 1
    out["label_anchor"] = (
        (out["label_accept"] == 1) & (out["label_position"] == 1)
    ).astype(int)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

# Columns that are IDs, labels, or free-text — never used as features
_EXCLUDE_COLS = {
    "session_id", "user_id", "restaurant_id", "item_id",
    "label_accept", "label_position", "label_cart_abandoned",
    "final_order_value", "label_aov_proxy", "event_timestamp",
    "label_aov", "label_timing", "label_anchor",
    "rest_name", "rest_cuisine_tags", "rest_discount_thresholds",
    "user_veg_days", "session_start",
}

_CATEGORICAL_COLS = [
    "user_segment", "user_city", "user_dietary_preference",
    "item_category", "item_subcategory", "item_cuisine_tag",
    "rest_primary_cuisine", "rest_city", "rest_zone", "rest_price_tier",
    "peak_hour_mode", "season_bucket",
]

_BOOL_COLS = ["item_veg_flag", "item_bestseller_flag", "item_availability"]


def prepare_features(
    df: pd.DataFrame,
    gru_hidden: np.ndarray,
) -> Tuple[pd.DataFrame, List[str], Dict[str, LabelEncoder]]:
    """Append GRU features, encode categoricals, return feature-col list."""

    out = df.copy()

    # Append 64 GRU hidden-state features
    for i in range(gru_hidden.shape[1]):
        out[f"gru_h_{i}"] = gru_hidden[:, i]

    # Encode categoricals
    encoders: Dict[str, LabelEncoder] = {}
    for col in _CATEGORICAL_COLS:
        if col in out.columns:
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].astype(str))
            encoders[col] = le

    # Booleans → int
    for col in _BOOL_COLS:
        if col in out.columns:
            out[col] = out[col].astype(int)

    feature_cols = [c for c in out.columns if c not in _EXCLUDE_COLS]
    return out, feature_cols, encoders


# ═══════════════════════════════════════════════════════════════════════════
# TEMPORAL SPLIT
# ═══════════════════════════════════════════════════════════════════════════


def temporal_split(
    df: pd.DataFrame,
    sessions: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split by session start date to avoid leaking future data.

    Train : weeks 49-51 (< Dec 22)
    Val   : week 52     (Dec 22 – 28)
    Test  : weeks 1-2   (≥ Dec 29)
    """
    sess_start = pd.to_datetime(
        sessions.set_index("session_id")["start_time"]
    )
    feat_start = df["session_id"].map(sess_start)
    df["session_start"] = feat_start  # keep for later grouping

    train_mask = (feat_start < "2025-12-22").values
    val_mask = (
        (feat_start >= "2025-12-22") & (feat_start < "2025-12-29")
    ).values
    test_mask = (feat_start >= "2025-12-29").values

    return train_mask, val_mask, test_mask


# ═══════════════════════════════════════════════════════════════════════════
# QUERY GROUPS (for LambdaRank)
# ═══════════════════════════════════════════════════════════════════════════


def _build_query_groups(df_subset: pd.DataFrame) -> np.ndarray:
    """Return array of group sizes for LambdaRank, ordered by session_id.

    The data must already be sorted by session_id (it is, from Step 4).
    """
    return df_subset.groupby("session_id", sort=False).size().values


# ═══════════════════════════════════════════════════════════════════════════
# MODEL CONFIGS
# ═══════════════════════════════════════════════════════════════════════════

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "accept": {
        "label": "label_accept",
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [8],
    },
    "aov": {
        "label": "label_aov",
        "objective": "regression",
        "metric": "rmse",
    },
    "abandon": {
        "label": "label_cart_abandoned",
        "objective": "binary",
        "metric": "auc",
    },
    "timing": {
        "label": "label_timing",
        "objective": "binary",
        "metric": "auc",
    },
    "anchor": {
        "label": "label_anchor",
        "objective": "binary",
        "metric": "auc",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════


def _default_lgb_params() -> Dict[str, Any]:
    return {
        "num_leaves": 63,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }


def train_lgbm_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    params_override: Optional[Dict[str, Dict[str, Any]]] = None,
    num_boost_round: int = 500,
) -> Dict[str, lgb.Booster]:
    """Train all five LightGBM models."""

    models: Dict[str, lgb.Booster] = {}
    X_train = df.loc[train_mask, feature_cols]
    X_val = df.loc[val_mask, feature_cols]

    for name, config in MODEL_CONFIGS.items():
        print(f"\n  Training [{name}] ({config['objective']}) ...")

        y_train = df.loc[train_mask, config["label"]].values
        y_val = df.loc[val_mask, config["label"]].values

        lgb_params = _default_lgb_params()
        lgb_params["objective"] = config["objective"]
        lgb_params["metric"] = config["metric"]

        if params_override and name in params_override:
            lgb_params.update(params_override[name])

        if config["objective"] == "lambdarank":
            lgb_params["eval_at"] = config.get("eval_at", [8])
            train_groups = _build_query_groups(df.loc[train_mask])
            val_groups = _build_query_groups(df.loc[val_mask])
            dtrain = lgb.Dataset(X_train, y_train, group=train_groups)
            dval = lgb.Dataset(X_val, y_val, group=val_groups, reference=dtrain)
        else:
            dtrain = lgb.Dataset(X_train, y_train)
            dval = lgb.Dataset(X_val, y_val, reference=dtrain)

        callbacks = [
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(100),
        ]

        booster = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=callbacks,
        )

        models[name] = booster
        print(f"  [{name}] best iteration = {booster.best_iteration}")

    return models


# ═══════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════════════

_TUNE_GRID = {
    "num_leaves": [31, 63],
    "learning_rate": [0.03, 0.1],
    "min_child_samples": [20, 50],
}


def tune_hyperparameters(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    model_name: str = "accept",
    num_boost_round: int = 200,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Grid search over key hyperparameters for a single model."""

    config = MODEL_CONFIGS[model_name]
    X_train = df.loc[train_mask, feature_cols]
    X_val = df.loc[val_mask, feature_cols]
    y_train = df.loc[train_mask, config["label"]].values
    y_val = df.loc[val_mask, config["label"]].values

    is_rank = config["objective"] == "lambdarank"
    if is_rank:
        train_groups = _build_query_groups(df.loc[train_mask])
        val_groups = _build_query_groups(df.loc[val_mask])

    # Pre-build datasets once (LightGBM reuses internal data)
    if is_rank:
        dtrain = lgb.Dataset(X_train, y_train, group=train_groups, free_raw_data=False)
        dval = lgb.Dataset(X_val, y_val, group=val_groups, reference=dtrain, free_raw_data=False)
    else:
        dtrain = lgb.Dataset(X_train, y_train, free_raw_data=False)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain, free_raw_data=False)

    results: List[Dict[str, Any]] = []
    keys = list(_TUNE_GRID.keys())
    combos = list(iterproduct(*_TUNE_GRID.values()))
    print(f"  Tuning [{model_name}]: {len(combos)} combinations ...")

    for ci, combo in enumerate(combos):
        lgb_params = _default_lgb_params()
        lgb_params["objective"] = config["objective"]
        lgb_params["metric"] = config["metric"]
        for k, v in zip(keys, combo):
            lgb_params[k] = v
        if is_rank:
            lgb_params["eval_at"] = config.get("eval_at", [8])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            booster = lgb.train(
                lgb_params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dval],
                valid_names=["val"],
                callbacks=[lgb.early_stopping(20, verbose=False),
                           lgb.log_evaluation(0)],
            )

        best_score = booster.best_score["val"]
        metric_key = list(best_score.keys())[0]
        score = best_score[metric_key]

        row = dict(zip(keys, combo))
        row["best_iter"] = booster.best_iteration
        row["val_score"] = float(score)
        results.append(row)

        if (ci + 1) % 4 == 0:
            print(f"    {ci + 1}/{len(combos)} done")

    results_df = pd.DataFrame(results)

    # For lambdarank/ndcg higher is better; for rmse/binary lower is better
    ascending = config["objective"] != "lambdarank"
    results_df = results_df.sort_values("val_score", ascending=ascending)
    best = results_df.iloc[0]

    best_params = {k: int(best[k]) if k in ("num_leaves", "min_child_samples")
                   else float(best[k]) for k in keys}
    print(f"  Best: {best_params}  val_score={best['val_score']:.5f}")
    return best_params, results_df


# ═══════════════════════════════════════════════════════════════════════════
# BUSINESS SCORE
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "accept": 0.30,
    "aov": 0.30,
    "abandon": -0.20,
    "timing": 0.10,
    "anchor": 0.10,
}

# Peak-hour weight overrides
PEAK_WEIGHTS = {
    "lunch_C2O": {"accept": 0.20, "aov": 0.20, "abandon": -0.10,
                  "timing": 0.30, "anchor": 0.20},
    "dinner_AOV": {"accept": 0.25, "aov": 0.40, "abandon": -0.15,
                   "timing": 0.10, "anchor": 0.10},
    "late_night_impulse": {"accept": 0.35, "aov": 0.20, "abandon": -0.25,
                           "timing": 0.10, "anchor": 0.10},
}


def compute_business_score(
    preds: Dict[str, np.ndarray],
    peak_mode: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Weighted combination of all five model predictions.

    If *peak_mode* is provided (array of mode strings per row),
    weights are switched per-row according to the current time slot.
    """
    n = len(preds["accept"])
    scores = np.zeros(n, dtype=np.float64)

    if peak_mode is None:
        for key, w in DEFAULT_WEIGHTS.items():
            scores += w * preds[key]
    else:
        for i in range(n):
            mode = str(peak_mode[i])
            weights = PEAK_WEIGHTS.get(mode, DEFAULT_WEIGHTS)
            for key, w in weights.items():
                scores[i] += w * preds[key][i]

    return scores


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════


def _ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    rel = relevance[:k]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
    ideal = np.sort(rel)[::-1]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_models(
    models: Dict[str, lgb.Booster],
    df: pd.DataFrame,
    feature_cols: List[str],
    test_mask: np.ndarray,
    k: int = 8,
) -> Dict[str, Dict[str, float]]:
    """Per-model metrics on the test set."""

    X_test = df.loc[test_mask, feature_cols]
    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        preds = model.predict(X_test)
        config = MODEL_CONFIGS[name]
        label_col = config["label"]
        y_true = df.loc[test_mask, label_col].values
        metrics: Dict[str, float] = {}

        # AUC-ROC (for classification and ranking models)
        if name in ("accept", "abandon", "timing", "anchor"):
            try:
                metrics["auc_roc"] = float(roc_auc_score(y_true, preds))
            except ValueError:
                metrics["auc_roc"] = 0.0

        # Precision@k and NDCG@k (session-level, for accept model)
        if name == "accept":
            test_df = df.loc[test_mask].copy()
            test_df["_pred"] = preds
            p_at_k: List[float] = []
            n_at_k: List[float] = []

            for _, grp in test_df.groupby("session_id"):
                grp_sorted = grp.sort_values("_pred", ascending=False)
                top = grp_sorted["label_accept"].values[:k]
                p_at_k.append(float(top.sum()) / k)
                n_at_k.append(_ndcg_at_k(top.astype(float), k))

            metrics["precision_at_8"] = float(np.mean(p_at_k))
            metrics["ndcg_at_8"] = float(np.mean(n_at_k))

        if name == "aov":
            from sklearn.metrics import mean_squared_error
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, preds)))

        results[name] = metrics

    return results


def evaluate_by_segment(
    models: Dict[str, lgb.Booster],
    df: pd.DataFrame,
    feature_cols: List[str],
    test_mask: np.ndarray,
    segment_col: str = "user_segment",
    encoders: Optional[Dict[str, LabelEncoder]] = None,
    k: int = 8,
) -> pd.DataFrame:
    """Accept-model Precision@8 and NDCG@8 broken down by user segment."""

    accept_model = models["accept"]
    test_df = df.loc[test_mask].copy()
    X_test = test_df[feature_cols]
    test_df["_pred"] = accept_model.predict(X_test)

    # Decode segment labels back to names if encoder exists
    if encoders and segment_col in encoders:
        le = encoders[segment_col]
        test_df["_segment_name"] = le.inverse_transform(
            test_df[segment_col].astype(int)
        )
    else:
        test_df["_segment_name"] = test_df[segment_col].astype(str)

    rows: List[Dict[str, Any]] = []
    for seg_name, seg_df in test_df.groupby("_segment_name"):
        p_at_k: List[float] = []
        n_at_k: List[float] = []
        for _, grp in seg_df.groupby("session_id"):
            grp_sorted = grp.sort_values("_pred", ascending=False)
            top = grp_sorted["label_accept"].values[:k]
            p_at_k.append(float(top.sum()) / k)
            n_at_k.append(_ndcg_at_k(top.astype(float), k))

        rows.append({
            "segment": seg_name,
            "sessions": len(p_at_k),
            "precision_at_8": float(np.mean(p_at_k)),
            "ndcg_at_8": float(np.mean(n_at_k)),
        })

    return pd.DataFrame(rows).sort_values("segment")


def evaluate_by_cart_stage(
    models: Dict[str, lgb.Booster],
    df: pd.DataFrame,
    feature_cols: List[str],
    test_mask: np.ndarray,
    k: int = 8,
) -> pd.DataFrame:
    """Accept-model metrics broken down by cart stage (0–3)."""

    accept_model = models["accept"]
    test_df = df.loc[test_mask].copy()
    test_df["_pred"] = accept_model.predict(test_df[feature_cols])

    rows: List[Dict[str, Any]] = []
    for stage, stage_df in test_df.groupby("cart_stage"):
        p_at_k: List[float] = []
        n_at_k: List[float] = []
        for _, grp in stage_df.groupby("session_id"):
            grp_sorted = grp.sort_values("_pred", ascending=False)
            top = grp_sorted["label_accept"].values[:k]
            p_at_k.append(float(top.sum()) / k)
            n_at_k.append(_ndcg_at_k(top.astype(float), k))

        rows.append({
            "cart_stage": int(stage),
            "impressions": len(stage_df),
            "precision_at_8": float(np.mean(p_at_k)),
            "ndcg_at_8": float(np.mean(n_at_k)),
        })

    return pd.DataFrame(rows).sort_values("cart_stage")


# ═══════════════════════════════════════════════════════════════════════════
# COLD START EVALUATION
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_cold_start(
    models: Dict[str, lgb.Booster],
    df: pd.DataFrame,
    feature_cols: List[str],
    test_mask: np.ndarray,
    order_count_col: str = "user_order_count",
    cold_threshold: int = 5,
    k: int = 8,
) -> Dict[str, Dict[str, float]]:
    """Separate evaluation for cold-start (≤ threshold orders) vs warm users."""

    accept_model = models["accept"]
    test_df = df.loc[test_mask].copy()
    test_df["_pred"] = accept_model.predict(test_df[feature_cols])

    results: Dict[str, Dict[str, float]] = {}

    for label, mask_fn in [
        ("cold", lambda d: d[d[order_count_col] <= cold_threshold]),
        ("warm", lambda d: d[d[order_count_col] > cold_threshold]),
    ]:
        subset = mask_fn(test_df)
        if len(subset) == 0:
            results[label] = {"precision_at_8": 0.0, "ndcg_at_8": 0.0, "sessions": 0}
            continue

        p_at_k: List[float] = []
        n_at_k: List[float] = []
        for _, grp in subset.groupby("session_id"):
            grp_sorted = grp.sort_values("_pred", ascending=False)
            top = grp_sorted["label_accept"].values[:k]
            p_at_k.append(float(top.sum()) / k)
            n_at_k.append(_ndcg_at_k(top.astype(float), k))

        results[label] = {
            "precision_at_8": float(np.mean(p_at_k)),
            "ndcg_at_8": float(np.mean(n_at_k)),
            "sessions": len(p_at_k),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════


def shap_analysis(
    models: Dict[str, lgb.Booster],
    df: pd.DataFrame,
    feature_cols: List[str],
    test_mask: np.ndarray,
    save_dir: Path = MODEL_DIR,
    max_samples: int = 2000,
) -> Dict[str, Any]:
    """Run SHAP TreeExplainer on each model and save summary plots."""
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    X_test = df.loc[test_mask, feature_cols]

    if len(X_test) > max_samples:
        X_sample = X_test.sample(n=max_samples, random_state=42)
    else:
        X_sample = X_test

    shap_results: Dict[str, Any] = {}

    for name, model in models.items():
        print(f"  SHAP analysis for [{name}] ...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        shap_results[name] = {
            "shap_values": shap_values,
            "X_sample": X_sample,
        }

        # Summary plot
        plt.figure(figsize=(10, 8))
        # Filter out embedding and GRU features for readability
        non_emb_cols = [
            i for i, c in enumerate(feature_cols)
            if not c.startswith("item_emb_") and not c.startswith("gru_h_")
        ]
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values

        sv_filtered = sv[:, non_emb_cols]
        feat_names = [feature_cols[i] for i in non_emb_cols]
        shap.summary_plot(
            sv_filtered,
            X_sample.iloc[:, non_emb_cols],
            feature_names=feat_names,
            show=False,
            max_display=20,
        )
        plt.title(f"SHAP Feature Importance — {name.upper()} model")
        plt.tight_layout()
        plt.savefig(save_dir / f"shap_{name}.png", dpi=150)
        plt.close()
        print(f"    -> saved {save_dir / f'shap_{name}.png'}")

    return shap_results


# ═══════════════════════════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════


def save_models(models: Dict[str, lgb.Booster], save_dir: Path = MODEL_DIR):
    os.makedirs(save_dir, exist_ok=True)
    for name, booster in models.items():
        path = save_dir / f"lgbm_{name}.txt"
        booster.save_model(str(path))
        print(f"  Saved {path}")


def load_models(load_dir: Path = MODEL_DIR) -> Dict[str, lgb.Booster]:
    models: Dict[str, lgb.Booster] = {}
    for name in MODEL_CONFIGS:
        path = load_dir / f"lgbm_{name}.txt"
        if path.exists():
            models[name] = lgb.Booster(model_file=str(path))
    return models


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    print("CSAO Rail — Step 5b: LightGBM Multi-Objective Ranker")
    print("-" * 48)

    features = pd.read_csv(DATA_DIR / "training_features.csv")
    sessions = pd.read_csv(DATA_DIR / "sessions.csv")
    gru_hidden = np.load(DATA_DIR / "gru_hidden_states.npy")

    print(f"Features:     {len(features):,} rows")
    print(f"GRU hidden:   {gru_hidden.shape}")

    # Labels
    print("\nEngineering labels ...")
    features = engineer_labels(features)
    for lbl in ["label_accept", "label_aov", "label_cart_abandoned",
                "label_timing", "label_anchor"]:
        print(f"  {lbl}: mean={features[lbl].mean():.4f}")

    # Features
    print("\nPreparing features ...")
    features, feature_cols, encoders = prepare_features(features, gru_hidden)
    print(f"  {len(feature_cols)} feature columns")

    # Split
    train_mask, val_mask, test_mask = temporal_split(features, sessions)
    print(
        f"\nSplit: train={train_mask.sum():,}  val={val_mask.sum():,}  "
        f"test={test_mask.sum():,}"
    )

    # Train
    print("\n" + "=" * 48)
    print("TRAINING INITIAL MODELS")
    print("=" * 48)
    models = train_lgbm_models(features, feature_cols, train_mask, val_mask)

    # Hyperparameter tuning
    print("\n" + "=" * 48)
    print("HYPERPARAMETER TUNING")
    print("=" * 48)
    best_params_all: Dict[str, Dict[str, Any]] = {}
    for name in MODEL_CONFIGS:
        bp, _ = tune_hyperparameters(
            features, feature_cols, train_mask, val_mask, model_name=name
        )
        best_params_all[name] = bp

    # Retrain with best params
    print("\n" + "=" * 48)
    print("RETRAINING WITH BEST PARAMS")
    print("=" * 48)
    models = train_lgbm_models(
        features, feature_cols, train_mask, val_mask,
        params_override=best_params_all,
    )

    # Evaluate
    print("\n" + "=" * 48)
    print("EVALUATION")
    print("=" * 48)
    metrics = evaluate_models(models, features, feature_cols, test_mask)
    for name, m in metrics.items():
        print(f"  [{name}] {m}")

    print("\nBy segment:")
    seg_df = evaluate_by_segment(
        models, features, feature_cols, test_mask, encoders=encoders
    )
    print(seg_df.to_string(index=False))

    print("\nBy cart stage:")
    stage_df = evaluate_by_cart_stage(models, features, feature_cols, test_mask)
    print(stage_df.to_string(index=False))

    print("\nCold start analysis:")
    cold = evaluate_cold_start(models, features, feature_cols, test_mask)
    for label, m in cold.items():
        print(f"  [{label}] {m}")

    # SHAP
    print("\n" + "=" * 48)
    print("SHAP ANALYSIS")
    print("=" * 48)
    shap_analysis(models, features, feature_cols, test_mask)

    # Save
    save_models(models)
    print("\nDone.")


if __name__ == "__main__":
    main()
