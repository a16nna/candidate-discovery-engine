"""
validation.py — Internal output validation (mirrors validate_submission.py logic).

Run this before writing the final CSV to catch issues early.
"""

import re
import pandas as pd

REQUIRED_COLS = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")


def validate_output(df: pd.DataFrame) -> list[str]:
    """
    Validate the output DataFrame against hackathon submission rules.
    
    Returns a list of error strings (empty = valid).
    """
    errors = []

    # ── Column check ───────────────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return errors  # Can't check further without columns

    # ── Row count ──────────────────────────────────────────────────────────
    if len(df) != 100:
        errors.append(f"Must have exactly 100 rows; found {len(df)}.")

    # ── candidate_id format ────────────────────────────────────────────────
    for i, cid in enumerate(df["candidate_id"], start=2):
        cid_str = str(cid).strip()
        if not CANDIDATE_ID_PATTERN.match(cid_str):
            errors.append(f"Row {i}: candidate_id '{cid_str}' does not match CAND_XXXXXXX format.")

    # ── No duplicate candidate_ids ─────────────────────────────────────────
    if df["candidate_id"].nunique() != len(df):
        dupes = df[df["candidate_id"].duplicated()]["candidate_id"].tolist()
        errors.append(f"Duplicate candidate_ids: {dupes}")

    # ── Rank: 1–100, no gaps, no dupes ────────────────────────────────────
    ranks = df["rank"].tolist()
    if sorted(ranks) != list(range(1, 101)):
        errors.append(f"Ranks must be exactly 1–100 with no gaps or duplicates.")

    # ── Score is non-increasing by rank ───────────────────────────────────
    sorted_df = df.sort_values("rank")
    scores = sorted_df["score"].tolist()
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1]:
            errors.append(
                f"Score not non-increasing: rank {i + 1} score {scores[i]:.4f} "
                f"< rank {i + 2} score {scores[i + 1]:.4f}"
            )

    # ── Tie-break: equal scores sorted by candidate_id ascending ──────────
    for i in range(len(sorted_df) - 1):
        row_a = sorted_df.iloc[i]
        row_b = sorted_df.iloc[i + 1]
        if row_a["score"] == row_b["score"]:
            if row_a["candidate_id"] > row_b["candidate_id"]:
                errors.append(
                    f"Tie-break violation at ranks {int(row_a['rank'])}–{int(row_b['rank'])}: "
                    f"{row_a['candidate_id']} should come after {row_b['candidate_id']}."
                )

    # ── Reasoning is non-empty ─────────────────────────────────────────────
    empty_reasoning = df[df["reasoning"].isna() | (df["reasoning"].str.strip() == "")]
    if len(empty_reasoning) > 0:
        errors.append(
            f"{len(empty_reasoning)} rows have empty reasoning: "
            f"ranks {empty_reasoning['rank'].tolist()[:5]}..."
        )

    return errors
