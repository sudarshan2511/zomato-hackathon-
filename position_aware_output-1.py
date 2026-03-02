"""
Step 7 — Position-Aware Final Output

Takes the list of scored candidates (from Step 5 LightGBM + business score) and produces
the exact 8–10 items shown on the CSAO rail in the UI.

What Step 7 does:
-----------------
1. **Candidate cap** — At most 200 candidates (by business score) enter the final
   pipeline. This keeps latency and complexity bounded; the rest are discarded.

2. **Post-ranking (Step 6)** — The capped list is passed through the existing
   post-ranking layers: subcategory diversity, category mix, price shock, margin cap,
   time-of-day exclusions, and the distance-to-discount (D-t-D) override. Step 6 may
   already move a gap-closing item to position 1 when D-t-D conditions are met.

3. **Position 1** — The first slot is the “hook” that drives rail engagement:
   - If Step 6 applied a D-t-D override, position 1 is already that item (with
     nudge explanation). We leave it as is.
   - Otherwise, we set position 1 to the item with the **highest anchor_score**
     (the model that predicts “accepted when shown at position 1”) among the
     current rail, so the best position-1 item is always in slot 1.

4. **Positions 2–10** — All other slots stay ordered by **full business score**
   (accept, AOV, abandon, timing, anchor blend), so the rest of the rail is
   optimized for overall value and conversion.

5. **Explanation** — A one-line explanation for the position-1 item is set from
   the top contributing feature (e.g. “Complete your meal with a drink!” for
   beverage gap, or the D-t-D nudge message when override applied). This comes
   from generate_explanation() in post_ranking.

No new data or model files are required. Step 7 uses:
- Existing: training_features, sessions, menu, gru_hidden_states, lgbm_*.txt
- post_ranking.apply_post_ranking, post_ranking.generate_explanation
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from post_ranking import apply_post_ranking, generate_explanation

MAX_CANDIDATES_FOR_RANKING = 200
DEFAULT_RAIL_SIZE = 10


def position_aware_final_output(
    candidates: pd.DataFrame,
    cart_total: float,
    cart_size: int,
    peak_mode: str,
    rail_size: int = DEFAULT_RAIL_SIZE,
    max_candidates: int = MAX_CANDIDATES_FOR_RANKING,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Step 7: cap candidates → run Step 6 post-ranking → set position 1 to
    D-t-D override or best anchor; positions 2–rail_size by business score;
    set one-line explanation for position 1.

    Returns
    -------
    rail : DataFrame with columns rank, item_id, item_name, item_price, ...
    stats : dict with input, after_cap, output, dtd_override, ...
    """
    df = candidates.sort_values("business_score", ascending=False).reset_index(drop=True)
    n_input = len(df)
    if len(df) > max_candidates:
        df = df.head(max_candidates).reset_index(drop=True)
    n_after_cap = len(df)

    rail, stats = apply_post_ranking(df, cart_total, cart_size, peak_mode, rail_size=rail_size)
    stats["input"] = n_input
    stats["after_cap"] = n_after_cap

    if stats.get("dtd_override", 0) == 0 and "anchor_score" in rail.columns:
        idx = rail["anchor_score"].idxmax()
        rail = pd.concat(
            [
                rail.loc[[idx]],
                rail.drop(index=idx).sort_values("business_score", ascending=False),
            ],
            ignore_index=True,
        )
        rail["rank"] = range(1, len(rail) + 1)
        rail.loc[0, "explanation"] = generate_explanation(rail.iloc[0])

    return rail, stats
