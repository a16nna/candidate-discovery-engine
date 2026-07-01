"""
reasoning.py — Generate specific 1-2 sentence reasoning per candidate.
Uses the real schema fields for concreteness.
"""

from src.job_description import JD_RELEVANT_ASSESSMENTS


def generate_reasoning(candidate: dict, breakdown: dict) -> str:
    profile = candidate.get("profile", {}) or {}
    sig = candidate.get("redrob_signals", {}) or {}

    title = profile.get("current_title", "Candidate")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0) or 0
    location = profile.get("location", "")

    # Top skills (endorsed first)
    skills = sorted(
        candidate.get("skills", []) or [],
        key=lambda s: (s.get("endorsements", 0) or 0),
        reverse=True
    )
    top_skill_names = [s["name"] for s in skills[:3] if s.get("name")]

    # Assessment scores
    assessments = sig.get("skill_assessment_scores", {}) or {}
    jd_assessments = {k: v for k, v in assessments.items() if k in JD_RELEVANT_ASSESSMENTS}
    top_assessment = max(jd_assessments.items(), key=lambda x: x[1]) if jd_assessments else None

    # Active status
    from src.signals.behavioral import _days_since
    days_inactive = _days_since(sig.get("last_active_date", ""))
    active_str = (
        "active this week" if days_inactive <= 7 else
        "active this month" if days_inactive <= 30 else
        f"last active {days_inactive} days ago"
    )

    resp_rate = sig.get("recruiter_response_rate")
    open_flag = sig.get("open_to_work_flag", False)

    # --- Build primary sentence ---
    career_s = breakdown.get("career", 0)
    skills_s = breakdown.get("skills", 0)
    sem_s    = breakdown.get("semantic", 0)
    beh_s    = breakdown.get("behavioral", 1.0)

    company_str = f" at {company}" if company else ""
    loc_str = f" ({location})" if location else ""

    if career_s >= 0.7 and skills_s >= 0.3:
        skill_str = ", ".join(top_skill_names) if top_skill_names else "relevant ML skills"
        primary = (
            f"{title}{company_str} with {yoe:.1f}y experience; "
            f"strong fit on role and skills — {skill_str}."
        )
    elif sem_s >= 0.6 and career_s >= 0.5:
        primary = (
            f"{title}{company_str} ({yoe:.1f}y){loc_str}; "
            f"high semantic match to JD with relevant career history."
        )
    else:
        best_dim = max(
            [("career", career_s), ("skills", skills_s), ("semantic", sem_s)],
            key=lambda x: x[1]
        )
        primary = (
            f"{title}{company_str} ({yoe:.1f}y); "
            f"best signal on {best_dim[0]} ({best_dim[1]:.0%})."
        )

    # --- Build secondary sentence ---
    secondary_parts = []

    # Assessment proof
    if top_assessment and top_assessment[1] >= 60:
        secondary_parts.append(f"Platform-assessed {top_assessment[0]}: {top_assessment[1]:.0f}/100.")

    # Active / responsive
    if open_flag and days_inactive <= 30:
        secondary_parts.append(f"Open to work, {active_str}.")
    elif days_inactive > 90:
        secondary_parts.append(f"Caution: {active_str} — availability uncertain.")

    # Behavioral concern
    if beh_s < 0.80:
        if resp_rate is not None and float(resp_rate) < 0.3:
            secondary_parts.append(f"Low recruiter response rate ({float(resp_rate):.0%}) — may be hard to reach.")
        else:
            secondary_parts.append("Behavioral signals below average — review availability.")
    elif beh_s >= 1.1:
        secondary_parts.append("Strong platform signals: high response rate, recently active.")

    if secondary_parts:
        return f"{primary} {secondary_parts[0]}"
    return primary.strip()
