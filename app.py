"""
app.py — Streamlit sandbox for the Redrob hackathon.

Hosts the candidate ranker as a web interface.
Satisfies the submission_spec.md Section 10.5 sandbox requirement.

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud or HuggingFace Spaces for the sandbox link.
"""

import gzip
import json
import io
import time

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Candidate Discovery Engine",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Candidate Discovery Engine")
st.caption("INDIA RUNS Data & AI Challenge 2026 | Anna Mariya Martin")

st.markdown("""
This system ranks candidates for a job description using a **hybrid scoring pipeline**:
- **Semantic fit** (TF-IDF cosine similarity, 30%)
- **Skills fit** (required + endorsed, 25%)  
- **Career fit** (title relevance + trajectory, 20%)
- **Experience fit** (years in target range, 15%)
- **Education fit** (degree + field + institution, 10%)
- **Behavioral modifier** (endorsement trust × tenure consistency × engagement)
- **Honeypot detection** (impossible timelines, keyword stuffers, behavioral anomalies)
""")

st.divider()

# ── Input section ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📂 Upload Candidates")
    uploaded_file = st.file_uploader(
        "Upload candidates.jsonl or candidates.jsonl.gz",
        type=["jsonl", "gz", "json"],
        help="Upload a sample of candidates (works on small samples for the sandbox).",
    )

with col2:
    st.subheader("📋 Job Description")
    jd_text = st.text_area(
        "Paste job description here (or use the embedded default)",
        height=200,
        placeholder="Paste JD here, or leave blank to use the embedded JD...",
    )

top_n = st.slider("Number of top candidates to rank", min_value=10, max_value=100, value=20)

run_btn = st.button("🚀 Run Ranking", type="primary", use_container_width=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if uploaded_file is None:
        st.error("Please upload a candidates file first.")
        st.stop()

    with st.spinner("Loading candidates..."):
        content = uploaded_file.read()

        try:
            if uploaded_file.name.endswith(".gz"):
                with gzip.open(io.BytesIO(content), "rt", encoding="utf-8") as f:
                    candidates = [json.loads(line) for line in f if line.strip()]
            else:
                text = content.decode("utf-8")
                if text.strip().startswith("["):
                    candidates = json.loads(text)
                else:
                    candidates = [json.loads(line) for line in text.splitlines() if line.strip()]
        except Exception as e:
            st.error(f"Failed to parse candidates file: {e}")
            st.stop()

    st.info(f"Loaded **{len(candidates):,}** candidates.")

    # Sandbox cap: limit to 5,000 for reasonable runtime
    MAX_SANDBOX = 5_000
    if len(candidates) > MAX_SANDBOX:
        st.warning(
            f"Sandbox mode: running on first {MAX_SANDBOX:,} candidates "
            f"(full dataset runs via rank.py CLI)."
        )
        candidates = candidates[:MAX_SANDBOX]

    with st.spinner("Running ranking pipeline..."):
        try:
            from src.ranker.pipeline import RankingPipeline
            pipeline = RankingPipeline(
                jd_text=jd_text if jd_text.strip() else None,
                top_n=min(top_n, len(candidates)),
            )
            t0 = time.time()
            results_df = pipeline.run(candidates)
            elapsed = time.time() - t0
        except ImportError:
            st.error("Could not import ranking pipeline. Make sure you're running from the repo root.")
            st.stop()
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    st.success(f"Ranking complete in **{elapsed:.1f}s**!")

    # ── Results ───────────────────────────────────────────────────────────
    st.subheader(f"🏆 Top {len(results_df)} Candidates")

    # Score distribution
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Top Score", f"{results_df['score'].iloc[0]:.2f}")
    col_b.metric("Median Score", f"{results_df['score'].median():.2f}")
    col_c.metric("Candidates Ranked", len(results_df))

    # Display table
    display_df = results_df.copy()
    display_df["score"] = display_df["score"].round(2)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "candidate_id": "Candidate ID",
            "score": st.column_config.NumberColumn("Score (0-100)", format="%.2f"),
            "reasoning": "Reasoning",
        }
    )

    # Download button
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download submission.csv",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
    )

    # Score histogram
    st.subheader("📊 Score Distribution")
    st.bar_chart(results_df.set_index("rank")["score"])

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built for the INDIA RUNS Data & AI Challenge 2026. "
    "Architecture: TF-IDF semantic scoring + structured feature extraction + "
    "behavioral signal modifiers + honeypot detection. CPU-only, < 5 min on 100K candidates."
)
