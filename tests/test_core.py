"""
tests/test_core.py — Unit tests for all core components.
Uses the REAL candidate schema from sample_candidates.json.

Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.signals.honeypot import is_honeypot
from src.signals.behavioral import compute_behavioral_modifier
from src.ranker.features import (
    extract_skills_score,
    extract_career_score,
    extract_experience_score,
    extract_education_score,
    build_candidate_text,
)
from src.utils.validation import validate_output
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Real-schema candidate builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_candidate(candidate_id="CAND_0000001", title="ML Engineer",
                    yoe=6.0, company="Swiggy", industry="Food Delivery",
                    skills=None, career=None, education=None, signals=None):
    """Build a candidate using the REAL schema structure."""
    if skills is None:
        skills = [
            {"name": "Python", "proficiency": "advanced", "endorsements": 15, "duration_months": 60},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 30, "duration_months": 48},
            {"name": "Sentence Transformers", "proficiency": "intermediate", "endorsements": 10, "duration_months": 24},
            {"name": "Recommendation Systems", "proficiency": "advanced", "endorsements": 25, "duration_months": 60},
        ]
    if career is None:
        career = [{
            "company": company,
            "title": title,
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": int(yoe * 12),
            "is_current": True,
            "industry": industry,
            "company_size": "5001-10000",
            "description": "Built production embedding-based retrieval and ranking systems. Shipped sentence-transformers and FAISS-based vector search to real users. NDCG and A/B testing for evaluation.",
        }]
    if education is None:
        education = [{
            "institution": "IIT Bombay",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2014,
            "end_year": 2018,
            "grade": "8.5 CGPA",
            "tier": "tier_1",
        }]
    if signals is None:
        signals = {
            "profile_completeness_score": 90.0,
            "signup_date": "2025-06-01",
            "last_active_date": "2026-06-20",
            "open_to_work_flag": True,
            "profile_views_received_30d": 80,
            "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.85,
            "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {"FAISS": 75.0, "Recommendation Systems": 80.0},
            "connection_count": 500,
            "endorsements_received": 150,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30.0, "max": 55.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 35.0,
            "search_appearance_30d": 400,
            "saved_by_recruiters_30d": 12,
            "interview_completion_rate": 0.85,
            "offer_acceptance_rate": 0.6,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        }
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": f"{title} | Embeddings, Ranking & Retrieval",
            "summary": f"ML engineer with {yoe} years building production ranking and retrieval systems.",
            "location": "Hyderabad, Telangana",
            "country": "India",
            "years_of_experience": yoe,
            "current_title": title,
            "current_company": company,
            "current_company_size": "5001-10000",
            "current_industry": industry,
        },
        "career_history": career,
        "education": education,
        "skills": skills,
        "certifications": [],
        "languages": [],
        "redrob_signals": signals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Honeypot detection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHoneypotDetection:

    def test_clean_strong_candidate_passes(self):
        c = _make_candidate()
        flagged, reason = is_honeypot(c)
        assert not flagged, f"Strong ML candidate falsely flagged: {reason}"

    def test_future_start_date_flagged(self):
        c = _make_candidate(career=[{
            "company": "FutureCorp",
            "title": "Engineer",
            "start_date": "2030-01-01",
            "end_date": "2033-01-01",
            "duration_months": 36,
            "is_current": False,
            "industry": "Software",
        }])
        flagged, reason = is_honeypot(c)
        assert flagged, "Future start date should be flagged."

    def test_end_before_start_flagged(self):
        c = _make_candidate(career=[{
            "company": "TimeCorp",
            "title": "Engineer",
            "start_date": "2022-01-01",
            "end_date": "2019-01-01",
            "duration_months": -36,
            "is_current": False,
            "industry": "Software",
        }])
        flagged, reason = is_honeypot(c)
        assert flagged, "End before start should be flagged."

    def test_impossible_response_rate_flagged(self):
        c = _make_candidate(signals={
            **_make_candidate()["redrob_signals"],
            "recruiter_response_rate": 1.5,
        })
        flagged, reason = is_honeypot(c)
        assert flagged, "Response rate > 1.0 should be flagged."

    def test_extreme_keyword_stuffer_flagged(self):
        """55+ skills, zero endorsements, not in experience text."""
        skills = [{"name": f"Skill_{i}", "proficiency": "intermediate", "endorsements": 0, "duration_months": 6}
                  for i in range(55)]
        c = _make_candidate(
            skills=skills,
            career=[{
                "company": "Generic Corp",
                "title": "Engineer",
                "start_date": "2020-01-01",
                "end_date": "2024-01-01",
                "duration_months": 48,
                "is_current": False,
                "industry": "IT Services",
                "description": "Did various tasks.",  # none of the 55 skills mentioned
            }]
        )
        flagged, reason = is_honeypot(c)
        assert flagged, "55 skills with 0 endorsements and none in experience should flag."

    def test_nontechnical_title_with_ai_skills_flagged(self):
        """JD's explicit trap: Marketing Manager with all AI keywords."""
        ai_skills = [
            {"name": "Embeddings", "proficiency": "intermediate", "endorsements": 2, "duration_months": 6},
            {"name": "Pinecone", "proficiency": "beginner", "endorsements": 1, "duration_months": 3},
            {"name": "Fine-tuning LLMs", "proficiency": "beginner", "endorsements": 0, "duration_months": 2},
            {"name": "FAISS", "proficiency": "beginner", "endorsements": 1, "duration_months": 3},
            {"name": "Vector Search", "proficiency": "beginner", "endorsements": 0, "duration_months": 2},
            {"name": "LLM", "proficiency": "beginner", "endorsements": 0, "duration_months": 1},
        ]
        c = _make_candidate(
            title="Marketing Manager",
            skills=ai_skills,
            career=[
                {"company": "Wipro", "title": "Marketing Manager", "start_date": "2020-01-01",
                 "end_date": None, "duration_months": 48, "is_current": True,
                 "industry": "IT Services", "description": "Led marketing campaigns."},
                {"company": "TCS", "title": "Sales Executive", "start_date": "2018-01-01",
                 "end_date": "2019-12-31", "duration_months": 24, "is_current": False,
                 "industry": "IT Services", "description": "Managed client accounts."},
            ]
        )
        flagged, reason = is_honeypot(c)
        assert flagged, "Marketing Manager with AI keywords and no tech career should be flagged."

    def test_moderate_skill_count_not_flagged(self):
        """20 skills, some endorsements — normal profile."""
        skills = [{"name": f"Skill_{i}", "proficiency": "intermediate",
                   "endorsements": i % 5, "duration_months": 12}
                  for i in range(20)]
        c = _make_candidate(skills=skills)
        flagged, _ = is_honeypot(c)
        assert not flagged, "20 skills with some endorsements should not be flagged."


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction tests (real schema)
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureExtraction:

    def test_skills_score_strong_candidate(self):
        """Candidate with endorsed JD-relevant skills should score above 0.2."""
        c = _make_candidate()
        score = extract_skills_score(c)
        assert score > 0.2, f"Expected > 0.2, got {score}"

    def test_skills_score_ranks_better_candidate_higher(self):
        """More endorsed JD-relevant skills = higher score."""
        rich = _make_candidate(skills=[
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 40, "duration_months": 60},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 30, "duration_months": 48},
            {"name": "Pinecone", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
            {"name": "Sentence Transformers", "proficiency": "advanced", "endorsements": 25, "duration_months": 48},
            {"name": "Ranking", "proficiency": "advanced", "endorsements": 15, "duration_months": 36},
            {"name": "Recommendation Systems", "proficiency": "advanced", "endorsements": 35, "duration_months": 60},
        ])
        poor = _make_candidate(skills=[
            {"name": "Cooking", "proficiency": "intermediate", "endorsements": 5, "duration_months": 12},
        ])
        assert extract_skills_score(rich) > extract_skills_score(poor), \
            "Rich skill candidate should outscore poor match."

    def test_career_score_ml_engineer_high(self):
        """ML Engineer title should score > 0.6 on career."""
        c = _make_candidate(title="ML Engineer", industry="Food Delivery")
        score = extract_career_score(c)
        assert score > 0.6, f"Expected > 0.6 for ML Engineer, got {score}"

    def test_career_score_marketing_manager_low(self):
        """Marketing Manager should score < 0.15 (JD trap title)."""
        c = _make_candidate(
            title="Marketing Manager",
            career=[{"company": "Wipro", "title": "Marketing Manager",
                     "start_date": "2019-01-01", "end_date": None,
                     "duration_months": 60, "is_current": True,
                     "industry": "IT Services", "description": "Ran campaigns."}]
        )
        score = extract_career_score(c)
        assert score < 0.15, f"Expected < 0.15 for Marketing Manager, got {score}"

    def test_career_score_consulting_only_penalized(self):
        """Entire career at consulting firms should get < 0.5 multiplier."""
        c = _make_candidate(
            title="Software Engineer",
            career=[
                {"company": "TCS", "title": "Software Engineer", "start_date": "2019-01-01",
                 "end_date": "2022-01-01", "duration_months": 36, "is_current": False,
                 "industry": "IT Services", "description": "Did things."},
                {"company": "Infosys", "title": "Senior Engineer", "start_date": "2022-01-01",
                 "end_date": None, "duration_months": 24, "is_current": True,
                 "industry": "IT Services", "description": "Did more things."},
            ]
        )
        score = extract_career_score(c)
        # Product engineer would get 0.7 title + production bonus; consulting reduces it
        product_c = _make_candidate(title="Software Engineer", industry="Software")
        assert score < extract_career_score(product_c), \
            "Consulting-only career should score lower than product company."

    def test_experience_score_ideal_range(self):
        """6 years = ideal range → 1.0."""
        c = _make_candidate(yoe=6.0)
        score = extract_experience_score(c)
        assert score == 1.0, f"Expected 1.0 for 6 years (ideal range), got {score}"

    def test_experience_score_under_experienced(self):
        """2 years < 5 year minimum → < 0.6."""
        c = _make_candidate(yoe=2.0)
        score = extract_experience_score(c)
        assert score < 0.6, f"Expected < 0.6 for 2 years, got {score}"

    def test_experience_score_over_experienced_decay(self):
        """14+ years should score lower than 7 years for this JD."""
        c7 = _make_candidate(yoe=7.0)
        c14 = _make_candidate(yoe=14.0)
        assert extract_experience_score(c7) > extract_experience_score(c14), \
            "7 years should score higher than 14 years for 5-9 year JD."

    def test_education_tier1_high(self):
        """IIT B.Tech CS = tier_1 → should score > 0.75."""
        c = _make_candidate()
        score = extract_education_score(c)
        assert score > 0.75, f"Expected > 0.75 for tier_1 IIT B.Tech CS, got {score}"

    def test_education_tier3_lower(self):
        """tier_3 institution scores lower than tier_1."""
        tier1 = _make_candidate()
        tier3 = _make_candidate(education=[{
            "institution": "Some Private College",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2015,
            "end_year": 2019,
            "tier": "tier_3",
        }])
        assert extract_education_score(tier1) > extract_education_score(tier3), \
            "tier_1 should outscore tier_3."

    def test_build_candidate_text_non_empty(self):
        c = _make_candidate()
        text = build_candidate_text(c)
        assert len(text) > 50, "Candidate text should be substantial."
        assert "faiss" in text.lower() or "embedding" in text.lower(), \
            "JD-relevant skills should appear in text."


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral modifier tests (real signals)
# ─────────────────────────────────────────────────────────────────────────────

class TestBehavioralModifier:

    def test_no_signals_returns_neutral(self):
        c = {"candidate_id": "CAND_0000001", "redrob_signals": {}}
        mod = compute_behavioral_modifier(c)
        assert mod == 1.0

    def test_strong_signals_boost(self):
        """Active, open to work, high response rate → > 1.0."""
        c = _make_candidate()  # default signals are strong
        mod = compute_behavioral_modifier(c)
        assert mod > 1.0, f"Expected > 1.0 for strong signals, got {mod}"

    def test_inactive_candidate_penalized(self):
        """6+ months inactive, low response rate → < 1.0."""
        sig = {
            "profile_completeness_score": 85.0,
            "last_active_date": "2025-01-01",  # 17+ months ago
            "open_to_work_flag": False,
            "recruiter_response_rate": 0.05,  # JD flags < 10%
            "avg_response_time_hours": 300.0,
            "applications_submitted_30d": 0,
            "endorsements_received": 5,
            "notice_period_days": 120,
            "preferred_work_mode": "remote",
            "willing_to_relocate": False,
            "github_activity_score": -1,
            "saved_by_recruiters_30d": 0,
            "interview_completion_rate": 0.3,
            "offer_acceptance_rate": 0.2,
            "verified_email": False,
            "verified_phone": False,
            "linkedin_connected": False,
            "profile_views_received_30d": 0,
            "search_appearance_30d": 5,
            "skill_assessment_scores": {},
        }
        c = _make_candidate(signals=sig)
        mod = compute_behavioral_modifier(c)
        assert mod < 1.0, f"Expected < 1.0 for inactive candidate, got {mod}"

    def test_modifier_always_bounded(self):
        """Modifier must always be in [0.5, 1.25]."""
        for c in [_make_candidate(), _make_candidate(signals={})]:
            mod = compute_behavioral_modifier(c)
            assert 0.5 <= mod <= 1.25, f"Modifier out of bounds: {mod}"


# ─────────────────────────────────────────────────────────────────────────────
# Output validation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputValidation:

    def _valid_df(self):
        return pd.DataFrame({
            "candidate_id": [f"CAND_{i:07d}" for i in range(1, 101)],
            "rank":  list(range(1, 101)),
            "score": [100.0 - i * 0.5 for i in range(100)],
            "reasoning": [f"Good candidate #{i}" for i in range(100)],
        })

    def test_valid_passes(self):
        assert validate_output(self._valid_df()) == []

    def test_wrong_count_fails(self):
        errors = validate_output(self._valid_df().head(50))
        assert any("100" in e for e in errors)

    def test_duplicate_id_fails(self):
        df = self._valid_df()
        df.loc[5, "candidate_id"] = df.loc[0, "candidate_id"]
        errors = validate_output(df)
        assert any("uplicate" in e.lower() for e in errors)

    def test_non_monotonic_score_fails(self):
        df = self._valid_df()
        df.loc[0, "score"] = 5.0
        df.loc[1, "score"] = 90.0
        errors = validate_output(df)
        assert any("increasing" in e.lower() for e in errors)

    def test_empty_reasoning_fails(self):
        df = self._valid_df()
        df.loc[0, "reasoning"] = ""
        errors = validate_output(df)
        assert any("reasoning" in e.lower() for e in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
