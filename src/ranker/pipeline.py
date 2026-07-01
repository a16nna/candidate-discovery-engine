"""
pipeline.py — Ranking pipeline (updated for real schema).

Scoring weights calibrated against the actual JD:
  - Career fit is heavily weighted because the JD is explicit:
    "Marketing Manager with all AI keywords is not a fit"
  - Semantic fit captures the gap between what JD says vs means
  - Behavioral modifier is multiplicative — dead account = down-weighted regardless

Component    Weight   Signal
─────────────────────────────────────────────────────────
semantic      0.28    TF-IDF cosine: JD ↔ candidate text
skills        0.27    Required + nice-to-have with endorsement trust
career        0.25    Title relevance + consulting penalty + production ML
experience    0.12    Years vs 5-9 target (6-8 ideal)
education     0.08    Degree × tier × field
behavioral    ×mult   23 signals: availability, responsiveness, recency, proof
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.job_description import EMBEDDED_JD
from src.signals.behavioral import compute_behavioral_modifier
from src.signals.honeypot import is_honeypot
from src.ranker.features import (
    extract_skills_score,
    extract_career_score,
    extract_experience_score,
    extract_education_score,
    build_candidate_text,
)
from src.ranker.reasoning import generate_reasoning

logger = logging.getLogger(__name__)

WEIGHTS = {
    "semantic":    0.28,
    "skills":      0.27,
    "career":      0.25,
    "experience":  0.12,
    "education":   0.08,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


class RankingPipeline:
    def __init__(self, jd_text: Optional[str] = None, top_n: int = 100):
        self.jd_text = jd_text or EMBEDDED_JD
        self.top_n = top_n

    def _fit_tfidf(self, candidate_texts: list[str]):
        logger.info("Fitting TF-IDF vectorizer ...")
        all_texts = [self.jd_text] + candidate_texts
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25_000,
            sublinear_tf=True,
            min_df=2,
            stop_words="english",
        )
        matrix = self._vectorizer.fit_transform(all_texts)
        self._jd_vector = matrix[0]
        self._candidate_matrix = matrix[1:]
        logger.info("TF-IDF fitted.")

    def _semantic_scores(self) -> np.ndarray:
        sims = cosine_similarity(self._jd_vector, self._candidate_matrix).flatten()
        if sims.max() > 0:
            sims = sims / sims.max()
        return sims

    def run(self, candidates: list[dict]) -> pd.DataFrame:
        logger.info(f"Pipeline start: {len(candidates):,} candidates")

        # Stage 1: Honeypot filter
        logger.info("Stage 1: Honeypot detection ...")
        clean = []
        flagged_count = 0
        for c in candidates:
            hp, reason = is_honeypot(c)
            if hp:
                flagged_count += 1
                logger.debug(reason)
            else:
                clean.append(c)
        logger.info(f"  Flagged {flagged_count:,} honeypots. {len(clean):,} clean.")

        # Stage 2: Text representations
        logger.info("Stage 2: Building text representations ...")
        texts = [build_candidate_text(c) for c in clean]

        # Stage 3: Semantic scoring
        logger.info("Stage 3: TF-IDF semantic scoring ...")
        self._fit_tfidf(texts)
        sem_scores = self._semantic_scores()

        # Stage 4: Structured features
        logger.info("Stage 4: Structured feature scoring ...")
        skill_scores  = np.array([extract_skills_score(c)     for c in clean])
        career_scores = np.array([extract_career_score(c)     for c in clean])
        exp_scores    = np.array([extract_experience_score(c) for c in clean])
        edu_scores    = np.array([extract_education_score(c)  for c in clean])

        # Stage 5: Behavioral modifier
        logger.info("Stage 5: Behavioral signal modifiers ...")
        beh_mods = np.array([compute_behavioral_modifier(c) for c in clean])

        # Stage 6: Composite score
        logger.info("Stage 6: Computing composite scores ...")
        base = (
            WEIGHTS["semantic"]    * sem_scores   +
            WEIGHTS["skills"]      * skill_scores  +
            WEIGHTS["career"]      * career_scores +
            WEIGHTS["experience"]  * exp_scores    +
            WEIGHTS["education"]   * edu_scores
        )
        composite = np.clip(base * beh_mods, 0.0, 1.0)
        scores_100 = np.round(composite * 100, 4)

        # Stage 7: Sort, select top-N
        logger.info(f"Stage 7: Selecting top {self.top_n} ...")
        order = np.argsort(-scores_100, kind="stable")

        rows = []
        rank = 1
        for idx in order:
            if rank > self.top_n:
                break
            c = clean[idx]
            cid = c.get("candidate_id", f"CAND_{idx:07d}")
            breakdown = {
                "semantic":   round(float(sem_scores[idx]),  3),
                "skills":     round(float(skill_scores[idx]),  3),
                "career":     round(float(career_scores[idx]), 3),
                "experience": round(float(exp_scores[idx]),    3),
                "education":  round(float(edu_scores[idx]),    3),
                "behavioral": round(float(beh_mods[idx]),      3),
            }
            rows.append({
                "candidate_id": cid,
                "rank":         rank,
                "score":        float(scores_100[idx]),
                "reasoning":    generate_reasoning(c, breakdown),
            })
            rank += 1

        df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])

        # Tie-break: equal scores → candidate_id ascending
        df = df.sort_values(
            by=["score", "candidate_id"],
            ascending=[False, True],
        ).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        logger.info(
            f"Done. Score range: "
            f"{df['score'].iloc[0]:.2f} – {df['score'].iloc[-1]:.2f}"
        )
        return df
