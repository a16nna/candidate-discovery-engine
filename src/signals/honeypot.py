"""
honeypot.py — Honeypot detection using the REAL schema.

Known trap patterns from the JD and sample data:
  1. Title mismatch traps: "Project Manager / Marketing Manager / Accountant" with
     all AI keywords in skills — the JD explicitly mentions this.
  2. Pure consulting-only career: TCS/Infosys/Wipro/Accenture entire career.
  3. Inactive profiles: last_active_date > 6 months ago (JD: "not actually available").
  4. Impossible signal values: response_rate > 1.0, future dates, etc.
  5. Keyword stuffers: many skills, endorsements=0 on all, none in experience.
  6. Behavioral twins: near-duplicate signal profiles (caught by scoring naturally).

Conservative: only flag extreme/clear cases. One legitimate candidate falsely
excluded from top-100 = missed opportunity. Honeypot in top-100 = disqualification.
False negatives are SAFER than false positives.
"""

from datetime import datetime
from typing import Tuple

from src.job_description import PURE_CONSULTING_FIRMS

NOW = datetime.now().date()


def is_honeypot(candidate: dict) -> Tuple[bool, str]:
    """
    Returns (is_honeypot, reason). Only flag with HIGH confidence.
    """
    cid = candidate.get("candidate_id", "unknown")

    # Rule 1: Impossible behavioral signal values
    flag, reason = _check_impossible_signals(candidate)
    if flag:
        return True, f"[{cid}] {reason}"

    # Rule 2: Future dates in career history
    flag, reason = _check_impossible_dates(candidate)
    if flag:
        return True, f"[{cid}] {reason}"

    # Rule 3: Extreme keyword stuffer (very conservative threshold)
    flag, reason = _check_extreme_keyword_stuffer(candidate)
    if flag:
        return True, f"[{cid}] {reason}"

    # Rule 4: Non-technical title + ALL AI keywords in skills (JD explicit trap)
    flag, reason = _check_title_skill_mismatch_trap(candidate)
    if flag:
        return True, f"[{cid}] {reason}"

    return False, ""


def _check_impossible_signals(candidate: dict) -> Tuple[bool, str]:
    """Values that are mathematically impossible."""
    sig = candidate.get("redrob_signals", {}) or {}

    resp = sig.get("recruiter_response_rate")
    if resp is not None:
        try:
            if float(resp) > 1.0:
                return True, f"recruiter_response_rate > 1.0: {resp}"
        except (TypeError, ValueError):
            pass

    icr = sig.get("interview_completion_rate")
    if icr is not None:
        try:
            if float(icr) > 1.0:
                return True, f"interview_completion_rate > 1.0: {icr}"
        except (TypeError, ValueError):
            pass

    oar = sig.get("offer_acceptance_rate")
    if oar is not None:
        try:
            v = float(oar)
            if v > 1.0:
                return True, f"offer_acceptance_rate > 1.0: {oar}"
        except (TypeError, ValueError):
            pass

    views = sig.get("profile_views_received_30d")
    if views is not None:
        try:
            if int(views) > 50_000:
                return True, f"profile_views_received_30d implausible: {views}"
        except (TypeError, ValueError):
            pass

    return False, ""


def _parse_date(s) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _check_impossible_dates(candidate: dict) -> Tuple[bool, str]:
    """Dates that can't be real: future start, end before start, implausible tenure."""
    career = candidate.get("career_history", []) or []

    for h in career:
        start = _parse_date(h.get("start_date"))
        end = _parse_date(h.get("end_date"))

        if start and start.date() > NOW:
            return True, f"Career start in future: {h['start_date']}"

        if start and end and end < start:
            return True, f"End before start: {h.get('end_date')} < {h.get('start_date')}"

        dur = h.get("duration_months", 0) or 0
        if dur > 480:  # > 40 years in one role
            return True, f"Implausible duration: {dur} months at {h.get('company')}"

    for edu in candidate.get("education", []) or []:
        end_year = edu.get("end_year")
        if end_year:
            try:
                if int(end_year) > NOW.year + 5:
                    return True, f"Graduation year in far future: {end_year}"
            except (TypeError, ValueError):
                pass

    sig_date = candidate.get("redrob_signals", {}).get("signup_date", "")
    if sig_date:
        sig_parsed = _parse_date(sig_date)
        if sig_parsed and sig_parsed.date() > NOW:
            return True, f"signup_date in future: {sig_date}"

    return False, ""


def _check_extreme_keyword_stuffer(candidate: dict) -> Tuple[bool, str]:
    """
    Pure keyword stuffer: huge skill list, zero endorsements total,
    skills not mentioned anywhere in career descriptions.
    Very conservative — only catches extreme outliers.
    """
    skills = candidate.get("skills", []) or []
    if len(skills) < 40:
        return False, ""  # Low skill count = not a stuffer

    total_endorsements = sum(s.get("endorsements", 0) or 0 for s in skills)
    if total_endorsements > 0:
        return False, ""  # Has any endorsements = not flagging

    # Check if any skills appear in experience text
    exp_text = " ".join(
        (h.get("description", "") or "").lower()
        for h in (candidate.get("career_history", []) or [])
    )
    skill_names = [(s.get("name", "") or "").lower() for s in skills]
    in_exp = sum(1 for name in skill_names if name and name in exp_text)

    if len(skills) > 50 and total_endorsements == 0 and in_exp < 3:
        return True, (
            f"{len(skills)} skills listed, 0 endorsements, "
            f"only {in_exp} appear in experience descriptions."
        )

    return False, ""


def _check_title_skill_mismatch_trap(candidate: dict) -> Tuple[bool, str]:
    """
    The JD explicitly describes this trap:
    'A candidate who has all the AI keywords listed as skills but whose title
    is Marketing Manager is not a fit, no matter how perfect their skill list looks.'
    
    Only flag when: clearly non-tech title + many AI skills + no technical experience.
    """
    profile = candidate.get("profile", {}) or {}
    title = (profile.get("current_title", "") or "").lower()

    # Definitive non-technical titles
    non_tech_titles = [
        "marketing manager", "sales manager", "operations manager", "hr manager",
        "accountant", "customer support", "project manager", "business development",
        "civil engineer", "mechanical engineer", "content writer",
    ]

    if not any(nt in title for nt in non_tech_titles):
        return False, ""  # Not a non-tech title, skip

    # Has AI keywords in skills?
    skills = candidate.get("skills", []) or []
    skill_names = [(s.get("name", "") or "").lower() for s in skills]
    ai_skills = [
        "embedding", "vector", "pinecone", "faiss", "weaviate", "qdrant",
        "fine-tuning", "llm", "recommendation", "ranking", "retrieval",
        "sentence transformer", "nlp", "openai",
    ]
    ai_matches = sum(1 for ai in ai_skills if any(ai in sk for sk in skill_names))

    if ai_matches < 4:
        return False, ""  # Too few AI skills to be a planted trap

    # Has any technical career history?
    tech_industries = ["ai/ml", "software", "fintech", "e-commerce", "food delivery"]
    career = candidate.get("career_history", []) or []
    has_tech_career = any(
        any(ti in (h.get("industry", "") or "").lower() for ti in tech_industries)
        for h in career
    )

    if has_tech_career:
        return False, ""  # Has actual tech experience — not a trap

    # Non-tech title + AI skills + no tech career = explicit JD trap
    return True, (
        f"Non-technical title '{profile.get('current_title')}' with "
        f"{ai_matches} AI keyword skills but no technical career history."
    )
