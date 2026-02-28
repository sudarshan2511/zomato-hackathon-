"""
CSAO Rail Recommendation System — Step 5a: GRU Cart Encoder

Trains a small GRU that processes cart items in addition order and
produces a 64-dim hidden state capturing the sequential meal trajectory.
"Biryani then Salan" encodes differently than "Salan alone."

The hidden state is extracted after training and used as 64 numeric
features fed directly into LightGBM alongside all other features.

Inputs:
    data/training_features.csv   (from Step 4)
    data/cart_events.csv         (from Step 1)
    data/item_embeddings.npz     (from Step 3)

Outputs:
    models/gru_encoder.pt        (trained GRU weights)
    data/gru_hidden_states.npy   (N × 64 matrix aligned with training_features)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CartGRUEncoder(nn.Module):
    """Single-layer GRU that maps a variable-length cart sequence to a
    64-dim trajectory vector."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        padded_seq: torch.Tensor,
        lengths: torch.Tensor,
        candidate_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        packed = pack_padded_sequence(
            padded_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        hidden = hidden.squeeze(0)  # (batch, hidden_dim)
        combined = torch.cat([hidden, candidate_emb], dim=1)
        logits = self.head(combined).squeeze(-1)
        return logits, hidden

    def encode(
        self, padded_seq: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Extract hidden state only (no classifier head)."""
        packed = pack_padded_sequence(
            padded_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return hidden.squeeze(0)


# ---------------------------------------------------------------------------
# Dataset / DataLoader helpers
# ---------------------------------------------------------------------------


class CartSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: List[np.ndarray],
        candidate_embs: np.ndarray,
        labels: np.ndarray,
    ):
        self.sequences = sequences
        self.candidate_embs = torch.from_numpy(candidate_embs).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = torch.from_numpy(self.sequences[idx]).float()
        return seq, self.candidate_embs[idx], self.labels[idx]


def _collate_cart_sequences(batch):
    sequences, candidates, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    return padded, lengths, torch.stack(candidates), torch.stack(labels)


# ---------------------------------------------------------------------------
# Embedding loader
# ---------------------------------------------------------------------------


def load_embeddings(
    path: Path = DATA_DIR / "item_embeddings.npz",
) -> Dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    ids = data["item_ids"]
    vecs = data["embeddings"]
    return {str(iid): vecs[i] for i, iid in enumerate(ids)}


# ---------------------------------------------------------------------------
# Cart-sequence construction
# ---------------------------------------------------------------------------


def build_cart_sequences(
    features_df: pd.DataFrame,
    cart_events_df: pd.DataFrame,
    emb_lookup: Dict[str, np.ndarray],
    emb_dim: int = 384,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Reconstruct the cart item sequence at each recommendation impression.

    For every row in *features_df* (a recommendation impression), finds all
    items that were in the cart *before* that impression (organic adds +
    accepted earlier recommendations) and maps them to their embedding
    vectors.

    Returns
    -------
    sequences : list of (seq_len, emb_dim) ndarrays — one per row.
        Empty carts are represented as a single zero-vector so that
        ``pack_padded_sequence`` always sees length >= 1.
    candidate_embs : (N, emb_dim) ndarray
    labels : (N,) ndarray of 0/1 accept labels
    """
    zero_emb = np.zeros(emb_dim, dtype=np.float32)

    ce = cart_events_df.copy()
    ce["timestamp"] = pd.to_datetime(ce["timestamp"])
    events_by_session: Dict[str, pd.DataFrame] = {
        sid: grp.sort_values("timestamp").reset_index(drop=True)
        for sid, grp in ce.groupby("session_id")
    }

    feat = features_df.copy()
    feat["event_timestamp"] = pd.to_datetime(feat["event_timestamp"])
    feat = feat.reset_index(drop=True)

    n = len(feat)
    sequences: List[np.ndarray] = [None] * n  # type: ignore[list-item]
    candidate_embs = np.zeros((n, emb_dim), dtype=np.float32)
    labels = np.zeros(n, dtype=np.float32)

    processed = 0
    for session_id, grp in feat.groupby("session_id"):
        sess_events = events_by_session.get(session_id)

        for idx in grp.index:
            ts = feat.at[idx, "event_timestamp"]
            item_id = str(feat.at[idx, "item_id"])

            cart_ids: List[str] = []
            if sess_events is not None and len(sess_events) > 0:
                past = sess_events[sess_events["timestamp"] < ts]
                in_cart = past[
                    (past["was_recommendation"] == False)  # noqa: E712
                    | (past["was_accepted"] == True)  # noqa: E712
                ]
                cart_ids = in_cart["item_id"].astype(str).tolist()

            if len(cart_ids) == 0:
                sequences[idx] = zero_emb.reshape(1, emb_dim)
            else:
                sequences[idx] = np.array(
                    [emb_lookup.get(iid, zero_emb) for iid in cart_ids],
                    dtype=np.float32,
                )

            candidate_embs[idx] = emb_lookup.get(item_id, zero_emb)
            labels[idx] = float(feat.at[idx, "label_accept"])

        processed += len(grp)
        if processed % 20_000 < len(grp):
            print(
                f"  Sequences built: {processed:,}/{n:,} "
                f"({100 * processed / n:.0f}%)"
            )

    print(f"  Sequences built: {n:,}/{n:,} (100%)")
    return sequences, candidate_embs, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_gru(
    sequences: List[np.ndarray],
    candidate_embs: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    input_dim: int = 384,
    hidden_dim: int = 64,
    batch_size: int = 256,
    lr: float = 1e-3,
    epochs: int = 20,
    patience: int = 3,
    device: str = "cpu",
) -> Tuple[CartGRUEncoder, Dict[str, List[float]]]:
    """Train GRU cart encoder with early stopping on validation BCE loss."""

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    train_ds = CartSequenceDataset(
        [sequences[i] for i in train_idx],
        candidate_embs[train_idx],
        labels[train_idx],
    )
    val_ds = CartSequenceDataset(
        [sequences[i] for i in val_idx],
        candidate_embs[val_idx],
        labels[val_idx],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_cart_sequences,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_cart_sequences,
    )

    model = CartGRUEncoder(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # --- train ---
        model.train()
        batch_losses = []
        for padded, lengths, cands, labs in train_loader:
            padded = padded.to(device)
            cands = cands.to(device)
            labs = labs.to(device)
            optimizer.zero_grad()
            logits, _ = model(padded, lengths, cands)
            loss = criterion(logits, labs)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        # --- validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for padded, lengths, cands, labs in val_loader:
                padded = padded.to(device)
                cands = cands.to(device)
                labs = labs.to(device)
                logits, _ = model(padded, lengths, cands)
                val_losses.append(criterion(logits, labs).item())

        avg_train = float(np.mean(batch_losses))
        avg_val = float(np.mean(val_losses))
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        print(
            f"  Epoch {epoch + 1:>2}/{epochs}  "
            f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------


def extract_hidden_states(
    model: CartGRUEncoder,
    sequences: List[np.ndarray],
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Run trained GRU on every sequence and return (N, 64) hidden states."""

    model.eval()
    model.to(device)

    n = len(sequences)
    hidden_dim = model.hidden_dim
    all_hidden = np.zeros((n, hidden_dim), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_seqs = [
            torch.from_numpy(sequences[i]).float() for i in range(start, end)
        ]
        lengths = torch.tensor([len(s) for s in batch_seqs], dtype=torch.long)
        padded = pad_sequence(batch_seqs, batch_first=True).to(device)

        with torch.no_grad():
            hidden = model.encode(padded, lengths)

        all_hidden[start:end] = hidden.cpu().numpy()

        if (start // batch_size) % 20 == 0:
            print(f"  Extracted: {start:,}/{n:,} ({100 * start / n:.0f}%)")

    print(f"  Extracted: {n:,}/{n:,} (100%)")
    return all_hidden


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> Tuple[CartGRUEncoder, np.ndarray, Dict[str, List[float]]]:
    print("CSAO Rail — Step 5a: GRU Cart Encoder")
    print("-" * 48)

    features = pd.read_csv(DATA_DIR / "training_features.csv")
    cart_events = pd.read_csv(DATA_DIR / "cart_events.csv")
    emb_lookup = load_embeddings()
    emb_dim = next(iter(emb_lookup.values())).shape[0]

    print(f"Features:    {len(features):,} rows")
    print(f"Cart events: {len(cart_events):,} rows")
    print(f"Embedding dim: {emb_dim}")

    print("\nBuilding cart sequences ...")
    sequences, candidate_embs, labels = build_cart_sequences(
        features, cart_events, emb_lookup, emb_dim
    )

    # Temporal split (session-level dates ensure no session straddles splits)
    sessions = pd.read_csv(DATA_DIR / "sessions.csv")
    sess_start = pd.to_datetime(
        sessions.set_index("session_id")["start_time"]
    )
    feat_sess_start = features["session_id"].map(sess_start)
    train_mask = (feat_sess_start < "2025-12-22").values
    val_mask = (
        (feat_sess_start >= "2025-12-22") & (feat_sess_start < "2025-12-29")
    ).values

    print(f"\nTrain rows: {train_mask.sum():,}  |  Val rows: {val_mask.sum():,}")

    print("\nTraining GRU ...")
    model, history = train_gru(
        sequences,
        candidate_embs,
        labels,
        train_mask,
        val_mask,
        input_dim=emb_dim,
    )

    print("\nExtracting hidden states ...")
    hidden_states = extract_hidden_states(model, sequences)

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "gru_encoder.pt")
    np.save(DATA_DIR / "gru_hidden_states.npy", hidden_states)

    print(f"\nGRU weights  -> {MODEL_DIR / 'gru_encoder.pt'}")
    print(f"Hidden states -> {DATA_DIR / 'gru_hidden_states.npy'}  shape={hidden_states.shape}")

    return model, hidden_states, history


if __name__ == "__main__":
    main()
