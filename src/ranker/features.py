"""
features.py — Feature extraction using the REAL candidate schema.

Actual schema (from sample_candidates.json):
  candidate_id: str (CAND_XXXXXXX)
  profile: dict
    - current_title, headline, summary, location, country
    - years_of_experience (float), current_company, current_industry
  career_history: list of dicts
    - company, title, start_date, end_date, duration_months
    - is_current, industry, company_size, description
  education: list of dicts
    - institution, degree, field_of_study, start_year, end_year
    - grade, tier (tier_1 / tier_2 / tier_3 / tier_4)
  skills: list of dicts
    - name, proficiency, endorsements, duration_months
  certifications: list
  redrob_signals: dict (23 signals — see behavioral.py)
"""

from src.job_description import (
    REQUIRED_SKILLS, NICE_TO_HAVE_SKILLS, TARGET_TITLES,
    MIN_EXPERIENCE_YEARS, MAX_EXPERIENCE_YEARS, IDEAL_MIN_YEARS, IDEAL_MAX_YEARS,
    PURE_CONSULTING_FIRMS, PRODUCT_COMPANY_SIGNALS, PREFERRED_LOCATIONS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Text builder for TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_text(candidate: dict) -> str:
    """
    Flatten the candidate into a single string for TF-IDF.
    Weight important signals by repetition: title (3×), JD-skills (2×).
    """
    parts = []
    profile = candidate.get("profile", {}) or {}

    # Title and headline — most important (3×)
    title = profile.get("current_title", "") or ""
    headline = profile.get("headline", "") or ""
    for _ in range(3):
        if title:
            parts.append(title)
        if headline:
            parts.append(headline)

    # Summary
    summary = profile.get("summary", "") or ""
    if summary:
        parts.append(summary)

    # Skills — name + proficiency (2× for endorsed skills)
    for s in candidate.get("skills", []) or []:
        name = s.get("name", "") or ""
        if name:
            reps = 2 if s.get("endorsements", 0) > 0 else 1
            parts.extend([name] * reps)

    # Career history descriptions (very rich signal)
    for h in candidate.get("career_history", []) or []:
        parts.append(h.get("title", "") or "")
        parts.append(h.get("company", "") or "")
        desc = h.get("description", "") or ""
        if desc:
            parts.append(desc)

    # Education
    for e in candidate.get("education", []) or []:
        parts.append(e.get("degree", "") or "")
        parts.append(e.get("field_of_study", "") or "")
        parts.append(e.get("institution", "") or "")

    # Certifications
    for cert in candidate.get("certifications", []) or []:
        if isinstance(cert, str):
            parts.append(cert)
        elif isinstance(cert, dict):
            parts.append(cert.get("name", "") or "")

    return " ".join(p for p in parts if p.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Skills scoring
# ─────────────────────────────────────────────────────────────────────────────

def _get_candidate_skill_map(candidate: dict) -> dict[str, dict]:
    """Return {skill_name_lower: {endorsements, duration_months, proficiency}}"""
    result = {}
    for s in candidate.get("skills", []) or []:
        name = (s.get("name", "") or "").lower().strip()
        if name:
            result[name] = {
                "endorsements": s.get("endorsements", 0) or 0,
                "duration_months": s.get("duration_months", 0) or 0,
                "proficiency": s.get("proficiency", "") or "",
            }
    return result


def _skill_in_experience(skill_name: str, candidate: dict) -> bool:
    """Check if skill name appears in any career_history description."""
    name_lower = skill_name.lower()
    for h in candidate.get("career_history", []) or []:
        desc = (h.get("description", "") or "").lower()
        if name_lower in desc:
            return True
    return False


def _skill_match_score(jd_skill: str, skill_map: dict, candidate: dict) -> float:
    """
    Score how well a candidate matches a single JD skill.
    
    Returns 0 (not present) to 1.3 (endorsed + long duration + in experience).
    Anti-stuffing: skills with zero endorsements AND not in experience text
    get only 0.5 weight (present but unverified).
    """
    jd_lower = jd_skill.lower()

    # Find best match in skill_map (exact or partial)
    best_match = None
    for name, data in skill_map.items():
        if jd_lower == name or jd_lower in name or name in jd_lower:
            if best_match is None or data["endorsements"] > best_match["endorsements"]:
                best_match = data
                best_match["_name"] = name

    if best_match is None:
        return 0.0

    base = 1.0
    endorsements = best_match["endorsements"]
    duration = best_match["duration_months"]

    # Endorsement trust
    if endorsements >= 20:
        base *= 1.3
    elif endorsements >= 5:
        base *= 1.15
    elif endorsements > 0:
        base *= 1.05
    else:
        # Zero endorsements: check if it appears in experience
        if not _skill_in_experience(best_match["_name"], candidate):
            base *= 0.5  # unverified claim — penalize

    # Duration bonus (skill used for a long time = genuine expertise)
    if duration >= 48:
        base *= 1.1
    elif duration >= 24:
        base *= 1.05

    return min(base, 1.3)


def extract_skills_score(candidate: dict) -> float:
    """
    Weighted skill match score in [0, 1].
    Required skills: full weight. Nice-to-have: 0.35 weight.
    Catches keyword stuffers via the endorsement/experience verification.
    """
    skill_map = _get_candidate_skill_map(candidate)

    req_scores = [_skill_match_score(s, skill_map, candidate) for s in REQUIRED_SKILLS]
    nth_scores = [_skill_match_score(s, skill_map, candidate) * 0.35 for s in NICE_TO_HAVE_SKILLS]

    total = sum(req_scores) + sum(nth_scores)
    max_total = float(len(REQUIRED_SKILLS)) + float(len(NICE_TO_HAVE_SKILLS)) * 0.35

    # Also credit skill_assessment_scores as hard proof of skill
    from src.job_description import JD_RELEVANT_ASSESSMENTS
    assessments = (candidate.get("redrob_signals", {}) or {}).get("skill_assessment_scores", {}) or {}
    assessment_bonus = 0.0
    for k, v in assessments.items():
        if k in JD_RELEVANT_ASSESSMENTS and float(v) >= 50:
            assessment_bonus += 0.02 * (float(v) / 100)  # tiny bonus, up to 0.02 per skill

    return min((total / max_total) + assessment_bonus, 1.0) if max_total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Career scoring
# ─────────────────────────────────────────────────────────────────────────────

def _title_relevance(title: str) -> float:
    """How relevant is a title to the Senior AI Engineer role?"""
    if not title:
        return 0.0
    t = title.lower()

    # Perfect matches
    if any(tgt in t for tgt in [
        "recommendation", "search engineer", "ml engineer", "machine learning engineer",
        "ai engineer", "nlp engineer", "applied ml", "applied scientist",
        "ranking engineer", "retrieval engineer",
    ]):
        return 1.0

    # Strong matches
    if any(tgt in t for tgt in [
        "data scientist", "research engineer", "software engineer",
        "backend engineer", "data engineer", "platform engineer",
    ]):
        return 0.7

    # Weak matches — technical but not directly relevant
    if any(tgt in t for tgt in [
        "developer", "engineer", "analyst", "architect",
    ]):
        return 0.4

    # Traps: non-technical titles with AI keywords in skills
    if any(bad in t for bad in [
        "manager", "director", "vp ", "head of", "president",
        "accountant", "hr ", "sales", "marketing", "operations",
        "civil", "mechanical", "support", "customer"
    ]):
        return 0.05  # Very strong penalty — JD explicitly warns about this

    return 0.2


def _is_consulting_only(candidate: dict) -> bool:
    """
    Returns True if candidate's ENTIRE career is at pure consulting firms.
    JD explicitly says: "only worked at consulting firms in their entire career" → disqualifier.
    If they have ANY product company experience, this returns False.
    """
    career = candidate.get("career_history", []) or []
    if not career:
        return False

    product_exp_months = 0
    consulting_months = 0

    for h in career:
        company = (h.get("company", "") or "").lower()
        industry = (h.get("industry", "") or "").lower()
        duration = h.get("duration_months", 0) or 0

        is_consulting = any(firm in company for firm in PURE_CONSULTING_FIRMS)
        is_product = any(sig in industry for sig in PRODUCT_COMPANY_SIGNALS)

        if is_consulting and not is_product:
            consulting_months += duration
        else:
            product_exp_months += duration

    # Only flag as consulting-only if >90% of career is at consulting firms
    total = consulting_months + product_exp_months
    if total == 0:
        return False
    return (consulting_months / total) > 0.9


def _has_production_ml_experience(candidate: dict) -> bool:
    """
    Does the candidate have evidence of shipping ML systems to production?
    Looks for production/deployed signals in career descriptions.
    """
    production_keywords = [
        "production", "deployed", "shipped", "real users", "at scale",
        "serving", "inference", "a/b test", "online", "live system",
        "recommendation system", "ranking", "retrieval", "search",
        "embedding", "vector", "similarity", "mlops",
    ]
    for h in candidate.get("career_history", []) or []:
        desc = (h.get("description", "") or "").lower()
        matches = sum(1 for kw in production_keywords if kw in desc)
        if matches >= 2:
            return True
    return False


def extract_career_score(candidate: dict) -> float:
    """
    Career fit: title relevance + consulting-only penalty + production ML evidence.
    
    This is the decisive signal against the JD's explicit traps:
    - Project Manager with all AI keywords → 0.05 title score → low final
    - Marketing Manager → 0.05 title score → low final
    - Pure consulting career → 0.5× multiplier
    """
    profile = candidate.get("profile", {}) or {}
    current_title = profile.get("current_title", "") or ""

    title_score = _title_relevance(current_title)

    # Check career trajectory (most recent role = strongest signal)
    career = candidate.get("career_history", []) or []
    career_score = title_score

    # Production ML experience bonus
    if _has_production_ml_experience(candidate):
        career_score = min(career_score * 1.2 + 0.1, 1.0)

    # Consulting-only penalty (JD explicit disqualifier)
    if _is_consulting_only(candidate):
        career_score *= 0.5

    # Location bonus — JD prefers India, specifically Pune/Noida
    location = (profile.get("location", "") or "").lower()
    country = (profile.get("country", "") or "").lower()
    if any(loc in location for loc in PREFERRED_LOCATIONS) or "india" in country:
        career_score = min(career_score + 0.05, 1.0)

    # Notice period signal — JD says "love sub-30 day, can buy out up to 30"
    signals = candidate.get("redrob_signals", {}) or {}
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        career_score = min(career_score + 0.05, 1.0)
    elif notice > 90:
        career_score = max(career_score - 0.05, 0.0)

    return min(career_score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Experience scoring
# ─────────────────────────────────────────────────────────────────────────────

def extract_experience_score(candidate: dict) -> float:
    """
    Score years_of_experience from the profile dict (direct field, reliable).
    
    JD: 5–9 years, ideal 6–8. Peaks at 1.0 in the ideal range.
    """
    profile = candidate.get("profile", {}) or {}
    years = profile.get("years_of_experience")

    if years is None:
        # Fallback: sum duration_months from career_history
        career = candidate.get("career_history", []) or []
        total_months = sum(h.get("duration_months", 0) or 0 for h in career)
        years = total_months / 12.0

    try:
        years = float(years)
    except (TypeError, ValueError):
        return 0.2

    if years <= 0:
        return 0.1
    elif years < MIN_EXPERIENCE_YEARS:
        # Under-experienced: ramp to 0.6 at MIN_EXPERIENCE_YEARS
        return (years / MIN_EXPERIENCE_YEARS) * 0.6
    elif years <= IDEAL_MIN_YEARS:
        # In range but not ideal sweet spot
        return 0.75 + (years - MIN_EXPERIENCE_YEARS) / (IDEAL_MIN_YEARS - MIN_EXPERIENCE_YEARS) * 0.25
    elif years <= IDEAL_MAX_YEARS:
        # Sweet spot: 6–8 years
        return 1.0
    elif years <= MAX_EXPERIENCE_YEARS:
        # Still acceptable but towards upper bound
        return 0.9
    else:
        # Over-experienced (14.5 year PM with AI keywords should not score high)
        excess = years - MAX_EXPERIENCE_YEARS
        return max(0.55, 0.9 - excess * 0.06)


# ─────────────────────────────────────────────────────────────────────────────
# Education scoring
# ─────────────────────────────────────────────────────────────────────────────

# The schema has a tier field: tier_1, tier_2, tier_3, tier_4
TIER_SCORES = {
    "tier_1": 1.0,
    "tier_2": 0.8,
    "tier_3": 0.6,
    "tier_4": 0.4,
}

DEGREE_SCORES = {
    "ph.d": 1.0, "phd": 1.0, "doctorate": 1.0,
    "m.tech": 0.95, "mtech": 0.95, "m.e.": 0.9,
    "m.s.": 0.9, "ms": 0.9, "msc": 0.85, "master": 0.85,
    "b.tech": 0.8, "btech": 0.8, "b.e.": 0.8, "be": 0.8,
    "bachelor": 0.75, "bsc": 0.7, "b.sc": 0.7, "b.s.": 0.7,
    "diploma": 0.35,
}

RELEVANT_FIELDS = [
    "computer science", "computer engineering", "software engineering",
    "information technology", "data science", "artificial intelligence",
    "machine learning", "electronics", "electrical", "mathematics",
    "statistics", "physics", "information systems",
]


def extract_education_score(candidate: dict) -> float:
    """Score using the schema's actual tier field + degree + field of study."""
    education = candidate.get("education", []) or []
    if not education:
        return 0.3

    best = 0.0
    for edu in education:
        tier = edu.get("tier", "tier_3")
        degree = (edu.get("degree", "") or "").lower()
        field = (edu.get("field_of_study", "") or "").lower()

        tier_score = TIER_SCORES.get(tier, 0.5)

        deg_score = 0.5
        for key, val in DEGREE_SCORES.items():
            if key in degree:
                deg_score = val
                break

        field_score = 1.0 if any(f in field for f in RELEVANT_FIELDS) else 0.5

        combined = 0.5 * deg_score + 0.3 * tier_score + 0.2 * field_score
        best = max(best, combined)

    return min(best, 1.0)
