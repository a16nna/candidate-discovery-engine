"""
behavioral.py — Behavioral modifier using the REAL 23 redrob_signals fields.

Actual field names (from redrob_signals_doc.docx + sample_candidates.json):
  profile_completeness_score   0-100
  signup_date                  date string
  last_active_date             date string
  open_to_work_flag            bool
  profile_views_received_30d   int >= 0
  applications_submitted_30d   int >= 0
  recruiter_response_rate      0.0-1.0
  avg_response_time_hours      float >= 0
  skill_assessment_scores      dict[str, 0-100]
  connection_count             int >= 0
  endorsements_received        int >= 0
  notice_period_days           0-180
  expected_salary_range_inr_lpa  {min, max}
  preferred_work_mode          onsite/hybrid/remote/flexible
  willing_to_relocate          bool
  github_activity_score        -1 to 100 (-1 = no GitHub linked)
  search_appearance_30d        int >= 0
  saved_by_recruiters_30d      int >= 0
  interview_completion_rate    0.0-1.0
  offer_acceptance_rate        -1 to 1.0 (-1 = no prior offers)
  verified_email               bool
  verified_phone               bool
  linkedin_connected           bool
"""

from datetime import datetime, date

NOW = datetime.now().date()


def _safe_float(v, default=0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v, default=0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _days_since(date_str: str) -> int:
    """Days since a date string like '2026-05-24'. Returns 999 on parse error."""
    if not date_str:
        return 999
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (NOW - d).days
    except ValueError:
        return 999


def compute_behavioral_modifier(candidate: dict) -> float:
    """
    Compute a multiplicative modifier from the 23 redrob_signals.
    Range: [0.5, 1.25]
    
    Per the JD: "a perfect-on-paper candidate who hasn't logged in for 6 months
    and has a 5% recruiter response rate is, for hiring purposes, not actually available."
    """
    sig = candidate.get("redrob_signals", {}) or {}
    if not sig:
        return 1.0

    components = []

    # 1. AVAILABILITY (weight 0.25)
    avail = _availability_score(sig)
    components.append((avail, 0.25))

    # 2. RESPONSIVENESS (weight 0.25) — JD explicitly calls this out
    resp = _responsiveness_score(sig)
    components.append((resp, 0.25))

    # 3. RECENCY / ACTIVITY (weight 0.20)
    recency = _recency_score(sig)
    components.append((recency, 0.20))

    # 4. SOCIAL PROOF (weight 0.15)
    social = _social_proof_score(sig)
    components.append((social, 0.15))

    # 5. TRUST / VERIFICATION (weight 0.15)
    trust = _trust_score(sig)
    components.append((trust, 0.15))

    total_w = sum(w for _, w in components)
    weighted = sum(score * w for score, w in components) / total_w

    return max(0.5, min(1.25, weighted))


def _availability_score(sig: dict) -> float:
    """
    Is the candidate actually available?
    open_to_work_flag + notice_period_days + willing_to_relocate
    """
    score = 1.0

    # Open to work — strong positive
    if sig.get("open_to_work_flag"):
        score *= 1.2
    else:
        score *= 0.85

    # Notice period — JD says loves sub-30, can buy out 30, 30+ harder
    notice = _safe_int(sig.get("notice_period_days"), 90)
    if notice <= 0:
        score *= 1.1      # immediate joiner
    elif notice <= 30:
        score *= 1.1      # sweet spot
    elif notice <= 60:
        score *= 1.0      # acceptable
    elif notice <= 90:
        score *= 0.92     # stretching it
    else:
        score *= 0.80     # 3+ months notice — hard

    # Willing to relocate — JD prefers Pune/Noida
    if sig.get("willing_to_relocate"):
        score *= 1.05

    # Work mode — flexible/hybrid matches JD's hybrid-first culture
    mode = (sig.get("preferred_work_mode") or "").lower()
    if mode in ("flexible", "hybrid"):
        score *= 1.02
    elif mode == "remote":
        score *= 0.95  # JD has offices, prefers some in-person

    return min(score, 1.25)


def _responsiveness_score(sig: dict) -> float:
    """
    recruiter_response_rate + avg_response_time_hours + interview_completion_rate
    
    The JD explicitly flags: 5% response rate = not actually hireable.
    """
    score = 1.0

    # Recruiter response rate (most important signal here)
    resp_rate = _safe_float(sig.get("recruiter_response_rate"))
    if resp_rate >= 0.75:
        score *= 1.2
    elif resp_rate >= 0.5:
        score *= 1.05
    elif resp_rate >= 0.3:
        score *= 0.95
    elif resp_rate >= 0.1:
        score *= 0.80
    else:
        score *= 0.60  # < 10% — barely reachable

    # Response time (median hours)
    resp_time = _safe_float(sig.get("avg_response_time_hours"), 48)
    if resp_time <= 12:
        score *= 1.1
    elif resp_time <= 48:
        score *= 1.0
    elif resp_time <= 120:
        score *= 0.95
    else:
        score *= 0.88  # > 5 days to respond

    # Interview completion rate
    icr = _safe_float(sig.get("interview_completion_rate"), -1)
    if icr >= 0:  # -1 means no interview history
        if icr >= 0.85:
            score *= 1.1
        elif icr >= 0.6:
            score *= 1.0
        else:
            score *= 0.9  # drops interviews often

    return min(score, 1.25)


def _recency_score(sig: dict) -> float:
    """
    last_active_date + applications_submitted_30d
    
    6+ months inactive = not actually available per JD.
    """
    score = 1.0

    days_inactive = _days_since(sig.get("last_active_date", ""))

    if days_inactive <= 7:
        score *= 1.2    # Active this week
    elif days_inactive <= 30:
        score *= 1.1    # Active this month
    elif days_inactive <= 60:
        score *= 1.0    # Active recently
    elif days_inactive <= 90:
        score *= 0.90   # 2-3 months ago
    elif days_inactive <= 180:
        score *= 0.75   # 3-6 months — getting stale
    else:
        score *= 0.55   # 6+ months — "not actually available" per JD

    # Applications submitted: actively searching is a positive
    apps = _safe_int(sig.get("applications_submitted_30d"))
    if apps >= 5:
        score *= 1.08
    elif apps >= 2:
        score *= 1.03

    return min(score, 1.25)


def _social_proof_score(sig: dict) -> float:
    """
    endorsements_received + saved_by_recruiters_30d + profile_views_received_30d
    + github_activity_score
    """
    score = 1.0

    # Total endorsements received (platform-level)
    endorse = _safe_int(sig.get("endorsements_received"))
    if endorse >= 100:
        score *= 1.15
    elif endorse >= 50:
        score *= 1.08
    elif endorse >= 20:
        score *= 1.03
    elif endorse == 0:
        score *= 0.92

    # Saved by recruiters in last 30 days (strong positive — real demand)
    saved = _safe_int(sig.get("saved_by_recruiters_30d"))
    if saved >= 10:
        score *= 1.1
    elif saved >= 3:
        score *= 1.05
    elif saved == 0:
        score *= 0.97

    # GitHub activity (-1 = not linked; 0 = linked but inactive; >0 = active)
    github = _safe_float(sig.get("github_activity_score"), -1)
    if github >= 30:
        score *= 1.08   # active contributor
    elif github >= 10:
        score *= 1.03
    elif github == -1:
        score *= 0.97   # no GitHub linked (minor ding for AI role)
    elif github < 5:
        score *= 0.96   # linked but inactive

    return min(score, 1.2)


def _trust_score(sig: dict) -> float:
    """
    verified_email + verified_phone + linkedin_connected + profile_completeness_score
    """
    score = 1.0

    verified_email = sig.get("verified_email", False)
    verified_phone = sig.get("verified_phone", False)
    linkedin = sig.get("linkedin_connected", False)

    # Each verification adds trust
    verifications = sum([bool(verified_email), bool(verified_phone), bool(linkedin)])
    if verifications == 3:
        score *= 1.1
    elif verifications == 2:
        score *= 1.05
    elif verifications == 1:
        score *= 1.0
    else:
        score *= 0.88  # nothing verified — suspicious for a platform profile

    # Profile completeness
    completeness = _safe_float(sig.get("profile_completeness_score"))
    if completeness >= 90:
        score *= 1.05
    elif completeness >= 70:
        score *= 1.0
    elif completeness >= 50:
        score *= 0.95
    else:
        score *= 0.90

    return min(score, 1.15)
