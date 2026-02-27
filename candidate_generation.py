"""
CSAO Rail Recommendation System — Step 3: Candidate Generation

Offline:
    generate_and_save_embeddings()  →  encodes every menu item using
    Sentence Transformers (all-MiniLM-L6-v2) and persists the vectors
    to data/item_embeddings.npz.  Run once; never called at request time.

Runtime:
    CandidateGenerator  →  loads pre-computed embeddings, builds a query
    vector from the current cart via mean-pooling, scores filtered
    candidates with an exact numpy dot product, and returns the top-k.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
DATA_DIR = _ROOT / "data"
DEFAULT_EMBEDDINGS_PATH = DATA_DIR / "item_embeddings.npz"
DEFAULT_MENU_PATH = DATA_DIR / "menu_items.csv"


# ===================================================================
# PART 1 — Offline Embedding Generation
# ===================================================================

def generate_and_save_embeddings(
    menu_csv: str | Path = DEFAULT_MENU_PATH,
    output_path: str | Path = DEFAULT_EMBEDDINGS_PATH,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode every menu item and save embeddings to disk.

    Parameters
    ----------
    menu_csv : path to menu_items.csv
    output_path : destination .npz file
    model_name : Sentence Transformer model identifier
    batch_size : encoding batch size

    Returns
    -------
    (embeddings, item_ids) where embeddings has shape [n_items, dim]
    and item_ids is a 1-D array of matching item_id strings.
    """
    from sentence_transformers import SentenceTransformer

    menu = pd.read_csv(menu_csv)
    texts = (menu["name"] + ". " + menu["description"]).tolist()
    item_ids = menu["item_id"].values

    print(f"[Step 3] Encoding {len(texts)} items with {model_name} …")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        item_ids=item_ids,
    )
    print(f"[Step 3] Saved {embeddings.shape} embeddings -> {output_path}")
    return embeddings, item_ids


# ===================================================================
# PART 2 — Runtime Candidate Scorer
# ===================================================================

class CandidateGenerator:
    """Loads pre-computed item embeddings and scores filtered candidates
    against a cart-derived query vector using exact dot product search."""

    def __init__(self, embeddings_path: str | Path = DEFAULT_EMBEDDINGS_PATH):
        data = np.load(embeddings_path, allow_pickle=True)
        self._embeddings: np.ndarray = data["embeddings"]       # [n, dim]
        self._item_ids: np.ndarray = data["item_ids"]            # [n]
        self._id_to_idx: dict[str, int] = {
            iid: i for i, iid in enumerate(self._item_ids)
        }
        self._dim = self._embeddings.shape[1]
        self._model = None  # lazy-loaded only for empty-cart fallback

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def n_items(self) -> int:
        return len(self._item_ids)

    # ------------------------------------------------------------------
    # Query vector
    # ------------------------------------------------------------------
    def build_query_vector(
        self,
        cart_item_ids: list[str] | set[str],
        restaurant_cuisine: str | None = None,
    ) -> np.ndarray:
        """Mean-pool cart item embeddings into a single query vector.

        Falls back to encoding the restaurant cuisine name when the cart
        is empty (cold-start / empty-cart scenario).
        """
        ids = [iid for iid in cart_item_ids if iid in self._id_to_idx]

        if ids:
            indices = [self._id_to_idx[iid] for iid in ids]
            cart_vecs = self._embeddings[indices]
            query = cart_vecs.mean(axis=0)
        elif restaurant_cuisine:
            query = self._encode_text(restaurant_cuisine)
        else:
            return np.zeros(self._dim, dtype=np.float32)

        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        return query.astype(np.float32)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score_candidates(
        self,
        query_vector: np.ndarray,
        candidate_item_ids: list[str] | np.ndarray,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """Dot-product similarity between the query and each candidate.

        Returns a list of (item_id, score) tuples sorted descending,
        truncated to *top_k*.
        """
        idx_map = [
            (iid, self._id_to_idx[iid])
            for iid in candidate_item_ids
            if iid in self._id_to_idx
        ]
        if not idx_map:
            return []

        ids, indices = zip(*idx_map)
        candidate_matrix = self._embeddings[list(indices)]   # [m, dim]
        scores = candidate_matrix @ query_vector              # [m]

        order = np.argsort(scores)[::-1][:top_k]
        return [(ids[i], float(scores[i])) for i in order]

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def generate_candidates(
        self,
        cart_item_ids: list[str] | set[str],
        filtered_candidates_df: pd.DataFrame,
        restaurant_cuisine: str | None = None,
        top_k: int = 50,
    ) -> pd.DataFrame:
        """End-to-end: build query → score candidates → return ranked df.

        Parameters
        ----------
        cart_item_ids : items currently in the cart
        filtered_candidates_df : output of HardFilterPipeline.run_filters()
        restaurant_cuisine : fallback for empty-cart cold start
        top_k : number of candidates to return

        Returns
        -------
        DataFrame with the original menu columns plus a ``similarity_score``
        column, sorted descending by score.
        """
        query = self.build_query_vector(cart_item_ids, restaurant_cuisine)

        candidate_ids = filtered_candidates_df["item_id"].tolist()
        ranked = self.score_candidates(query, candidate_ids, top_k=top_k)

        if not ranked:
            return filtered_candidates_df.head(0)

        ranked_df = pd.DataFrame(ranked, columns=["item_id", "similarity_score"])
        result = ranked_df.merge(filtered_candidates_df, on="item_id", how="left")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _encode_text(self, text: str) -> np.ndarray:
        """Lazy-load the model only when needed (empty-cart fallback)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        vec = self._model.encode(
            [text], normalize_embeddings=True
        )
        return np.asarray(vec[0], dtype=np.float32)


# ===================================================================
# CLI — run embedding generation directly
# ===================================================================
if __name__ == "__main__":
    generate_and_save_embeddings()
