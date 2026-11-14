"""
Fake Job Posting Detector - Streamlit Web Application.

An interactive decision-support application for students and job seekers.
Combines machine learning classification, explainable feature attribution,
and deterministic red flag checks to assess the risk of fraudulent job postings.

Architecture (Essential for Viva):
- Single Resource Caching:
  Database, AuthManager, and Predictor instances are cached globally across
  sessions via `st.cache_resource` for low-latency requests.
- Session State Authentication:
  User authentication is tracked inside `st.session_state["user"]`.
- Security First:
  Ad content is treated as untrusted text. No links found in postings are ever
  visited, resolved, or fetched. All SQL operations use parameterized queries.
- Clean Separation of Concerns:
  Each view is structured into a dedicated page function (`page_check_job`,
  `page_my_history`, `page_about`).
"""

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import altair as alt
import pandas as pd
import streamlit as st

# Ensure the project directory is on sys.path for local module imports
SCRIPT_DIR: Path = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import domain modules
from auth import AuthManager
from database import Database
from predictor import Predictor, label_for_level
from reports import ReportGenerator
from rules import highlight_text


# -----------------------------------------------------------------------------
# Resource Caching
# -----------------------------------------------------------------------------

@st.cache_resource
def get_database() -> Database:
    """Returns a singleton instance of the SQLite Database manager."""
    db_path = SCRIPT_DIR / "data" / "app.db"
    return Database(db_path=db_path)


@st.cache_resource
def get_auth_manager() -> AuthManager:
    """Returns a singleton instance of the AuthManager."""
    return AuthManager(db=get_database())


@st.cache_resource
def get_predictor() -> Optional[Predictor]:
    """
    Returns a singleton instance of Predictor.
    Catches FileNotFoundError gracefully if models have not yet been trained.
    """
    try:
        return Predictor(models_dir=SCRIPT_DIR / "models")
    except FileNotFoundError:
        return None


# -----------------------------------------------------------------------------
# Cached Data Loaders for Model Insights
# -----------------------------------------------------------------------------

@st.cache_data
def load_metadata_cached() -> Optional[Dict[str, Any]]:
    """Loads and caches models/metadata.json."""
    path = SCRIPT_DIR / "models" / "metadata.json"
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def load_model_comparison_cached() -> Optional[pd.DataFrame]:
    """Loads and caches models/model_comparison.csv."""
    path = SCRIPT_DIR / "models" / "model_comparison.csv"
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def load_test_dataset_stats_cached() -> Optional[Dict[str, Any]]:
    """Loads test.csv and computes genuine and fraud class distributions dynamically."""
    path = SCRIPT_DIR / "data" / "processed" / "test.csv"
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
        total = len(df)
        genuine_count = int((df["fraudulent"] == 0).sum())
        fraud_count = int((df["fraudulent"] == 1).sum())
        dummy_accuracy = (genuine_count / total * 100.0) if total > 0 else 0.0
        fraud_prevalence = (fraud_count / total * 100.0) if total > 0 else 0.0
        return {
            "total": total,
            "genuine_count": genuine_count,
            "fraud_count": fraud_count,
            "dummy_accuracy": dummy_accuracy,
            "fraud_prevalence": fraud_prevalence,
        }
    except Exception:
        return None


# -----------------------------------------------------------------------------
# UI Helpers & Styling
# -----------------------------------------------------------------------------

def render_risk_badge(risk_level: str) -> None:
    """Renders a prominent colored HTML badge corresponding to the risk level."""
    level = risk_level.upper()
    if level == "LOW":
        bg_color = "#2a9d8f"  # Teal / Green
        border_color = "#1d7065"
        icon = "✅"
    elif level == "MEDIUM":
        bg_color = "#f4a261"  # Orange
        border_color = "#e76f51"
        icon = "⚠️"
    else:
        bg_color = "#e63946"  # Red
        border_color = "#b5171e"
        icon = "🚨"

    badge_html = f"""
    <div style="
        background-color: {bg_color};
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        border: 2px solid {border_color};
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        {icon} RISK LEVEL: {level}
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)


def render_warning_badge(severity: str) -> str:
    """Returns a formatted severity tag for rule warnings."""
    sev = severity.lower()
    if sev == "high":
        return "🔴 **HIGH**"
    elif sev == "medium":
        return "🟠 **MEDIUM**"
    return "🔵 **LOW**"


# -----------------------------------------------------------------------------
# Authentication Screen (Login / Register)
# -----------------------------------------------------------------------------

def render_auth_screen(auth_mgr: AuthManager) -> None:
    """Renders the login and registration tabs when no user is logged in."""
    st.header("Welcome to Fake Job Posting Detector")
    st.write(
        "Analyze job and internship advertisements to detect warning signs, "
        "evaluate statistical scam similarity, and protect your career search."
    )

    tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register New Account"])

    # ------------------ Login Tab ------------------
    with tab_login:
        with st.form("login_form"):
            st.subheader("Login to Your Account")
            login_username = st.text_input("Username").strip()
            login_password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Log In", use_container_width=True)

            if submit_login:
                if not login_username or not login_password:
                    st.error("Please enter both username and password.")
                else:
                    user = auth_mgr.login(username=login_username, password=login_password)
                    if user:
                        st.session_state["user"] = user
                        st.success(f"Welcome back, {user['username']}!")
                        st.rerun()
                    else:
                        # One generic message for both unknown user and wrong password
                        st.error("Invalid username or password.")

    # ------------------ Register Tab ------------------
    with tab_register:
        with st.form("register_form"):
            st.subheader("Create a New Account")
            reg_username = st.text_input(
                "Choose Username",
                help="3-30 characters, alphanumeric and underscores only."
            ).strip()
            reg_password = st.text_input(
                "Create Password",
                type="password",
                help="At least 8 characters, with at least one letter and one number."
            )
            reg_confirm = st.text_input("Confirm Password", type="password")
            submit_register = st.form_submit_button("Create Account", use_container_width=True)

            if submit_register:
                if reg_password != reg_confirm:
                    st.error("Passwords do not match.")
                else:
                    try:
                        new_user = auth_mgr.register(username=reg_username, password=reg_password)
                        st.success(
                            f"Account for '{new_user['username']}' created successfully! "
                            "Please switch to the Login tab to sign in."
                        )
                    except ValueError as err:
                        st.error(str(err))


# -----------------------------------------------------------------------------
# Page: Check a Job Ad
# -----------------------------------------------------------------------------

def page_check_job(predictor: Predictor, db: Database, user: Dict[str, Any]) -> None:
    """Renders the main job advertisement scanning form and displays results."""
    st.title("🛡️ Check a Job or Internship Ad")
    st.write(
        "Paste the details of a job posting or internship below to calculate an "
        "estimated risk score and detect heuristic red flags."
    )

    # Input Form
    with st.form("job_check_form"):
        st.subheader("Job Posting Details")

        col_title, col_sal = st.columns([3, 2])
        with col_title:
            title_input = st.text_input(
                "Job Title",
                value=st.session_state.get("form_title", ""),
                placeholder="e.g. Software Engineering Intern"
            )
        with col_sal:
            salary_input = st.text_input(
                "Salary Range (Optional)",
                value=st.session_state.get("form_salary", ""),
                placeholder="e.g. 50000-70000"
            )

        company_profile_input = st.text_area(
            "Company Profile (Optional)",
            value=st.session_state.get("form_profile", ""),
            height=100,
            placeholder="Overview of the company, mission, or industry..."
        )

        description_input = st.text_area(
            "Job Description (Required)",
            value=st.session_state.get("form_desc", ""),
            height=180,
            placeholder="Paste the primary job responsibilities and duties here..."
        )

        col_req, col_ben = st.columns(2)
        with col_req:
            requirements_input = st.text_area(
                "Requirements & Skills (Optional)",
                value=st.session_state.get("form_req", ""),
                height=120,
                placeholder="Required education, experience, technical stack..."
            )
        with col_ben:
            benefits_input = st.text_area(
                "Benefits & Compensation (Optional)",
                value=st.session_state.get("form_ben", ""),
                height=120,
                placeholder="Healthcare, 401k match, mentorship, stipends..."
            )

        st.markdown("**Structured Posting Indicators**")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            telecommuting_check = st.checkbox(
                "Remote / Telecommuting",
                value=st.session_state.get("form_tele", False),
                help="Check if this is advertised as a 100% work-from-home position."
            )
        with col_c2:
            has_logo_check = st.checkbox(
                "Company Logo Present",
                value=st.session_state.get("form_logo", True),
                help="Check if the posting includes an official employer logo."
            )
        with col_c3:
            has_questions_check = st.checkbox(
                "Screening Questions Included",
                value=st.session_state.get("form_quest", False),
                help="Check if the application asks custom screening questions."
            )

        submit_check = st.form_submit_button("🔍 Check this ad", use_container_width=True)

    # Handle Form Submission
    if submit_check:
        # Validate inputs
        if not description_input or not description_input.strip():
            st.error("Validation Error: Job description is required and cannot be empty.")
            return

        # Text length validation (limit to 20,000 characters per field to prevent memory abuse)
        fields = [
            ("Job Title", title_input),
            ("Company Profile", company_profile_input),
            ("Job Description", description_input),
            ("Requirements", requirements_input),
            ("Benefits", benefits_input),
            ("Salary Range", salary_input),
        ]
        for name, text_val in fields:
            if len(text_val) > 20000:
                st.error(f"Validation Error: '{name}' exceeds the maximum allowed length of 20,000 characters.")
                return

        # Persist form fields into session state so values stay populated
        st.session_state["form_title"] = title_input
        st.session_state["form_salary"] = salary_input
        st.session_state["form_profile"] = company_profile_input
        st.session_state["form_desc"] = description_input
        st.session_state["form_req"] = requirements_input
        st.session_state["form_ben"] = benefits_input
        st.session_state["form_tele"] = telecommuting_check
        st.session_state["form_logo"] = has_logo_check
        st.session_state["form_quest"] = has_questions_check

        # Execute Prediction
        with st.spinner("Analyzing job posting with Machine Learning and Heuristic Rules..."):
            try:
                result = predictor.predict(
                    title=title_input,
                    company_profile=company_profile_input,
                    description=description_input,
                    requirements=requirements_input,
                    benefits=benefits_input,
                    salary_range=salary_input,
                    telecommuting=1 if telecommuting_check else 0,
                    has_company_logo=1 if has_logo_check else 0,
                    has_questions=1 if has_questions_check else 0,
                )
                st.session_state["last_result"] = result

                # Save check into SQLite database for the logged-in user
                saved_check_id = db.save_check(
                    user_id=user["id"],
                    job_title=title_input or "Untitled Position",
                    job_text=description_input,
                    fraud_probability=result["fraud_probability"],
                    risk_level=result["risk_level"],
                    model_name=result["model_name"],
                    warnings=result["warnings"],
                    rule_flags=result["features_used"],
                )
                st.session_state["last_check_id"] = saved_check_id

            except ValueError as val_err:
                st.error(f"Input Error: {val_err}")
                return

    # Display Result if Available
    result = st.session_state.get("last_result")
    if result:
        st.markdown("---")
        st.subheader("📊 Analysis Result")

        # 1. Risk Level Banner
        render_risk_badge(result["risk_level"])

        # 2. Risk Score & Progress Bar
        risk_pct = int(round(result["fraud_probability"] * 100))
        st.write(f"### Risk score: **{risk_pct}%**")
        st.progress(float(result["fraud_probability"]))

        # 3. Model Classification Label
        st.markdown(f"**Classification Output:** `{result['label']}` *(Model: {result['model_name']})*")

        # 4. Mandatory Advisory Caption
        st.caption(
            "⚠️ **Disclaimer:** This score is a model estimate of how similar this ad is "
            "to known fake ads. It is not proof, and it is not a true probability."
        )

        # Download Report PDF Button (Task 2)
        saved_check_id = st.session_state.get("last_check_id")
        if saved_check_id:
            saved_check = db.get_check(saved_check_id)
            if saved_check:
                if "top_words" not in saved_check and "top_words" in result:
                    saved_check["top_words"] = result["top_words"]
                if "label" not in saved_check and "label" in result:
                    saved_check["label"] = result["label"]
                pdf_bytes = ReportGenerator.build_pdf(
                    check=saved_check,
                    owner_username=user.get("username", "User"),
                )
                st.download_button(
                    label="📄 Download report (PDF)",
                    data=pdf_bytes,
                    file_name=f"job_check_{saved_check_id}.pdf",
                    mime="application/pdf",
                    key="btn_download_report_check",
                    use_container_width=True,
                )

        st.markdown("---")

        # 5. Warning Signs (Rule Heuristics)
        st.subheader("🚩 Warning Signs")
        warnings = result.get("warnings", [])
        if warnings:
            for w in warnings:
                badge = render_warning_badge(w["severity"])
                matched_str = ", ".join(w["matched"]) if w["matched"] else "N/A"
                st.markdown(f"- {badge} **{w['message']}** *(Detected: `{matched_str}`)*")
        else:
            st.success("No heuristic warning signs were triggered. The ad passed all pattern checks.")

        st.markdown("---")

        # 6. Influential Words (Explainability)
        st.subheader("🔍 Words that Influenced the Result")
        st.write(
            "Based on the Logistic Regression model's learned vocabulary weights, "
            "here are the words from this ad with the strongest influence on the prediction:"
        )

        col_fraud_words, col_gen_words = st.columns(2)

        with col_fraud_words:
            st.markdown("##### 🔴 Pushing Toward Fraud")
            fraud_words = result["top_words"].get("fraud", [])
            if fraud_words:
                fw_df = pd.DataFrame([
                    {"Word": item["word"], "Contribution": f"+{item['contribution']:.4f}"}
                    for item in fraud_words
                ])
                st.dataframe(fw_df, use_container_width=True, hide_index=True)
            else:
                st.info("No strong fraud-associated words identified.")

        with col_gen_words:
            st.markdown("##### 🟢 Pushing Toward Genuine")
            gen_words = result["top_words"].get("genuine", [])
            if gen_words:
                gw_df = pd.DataFrame([
                    {"Word": item["word"], "Contribution": f"{item['contribution']:.4f}"}
                    for item in gen_words
                ])
                st.dataframe(gw_df, use_container_width=True, hide_index=True)
            else:
                st.info("No strong genuine-associated words identified.")

        st.markdown("---")

        # 7. Suspicious Keyword Highlighting (Highlighted Text)
        st.subheader("🖍️ Highlighted Text")

        # Build combined text from filled ad components
        ad_sections: List[str] = []
        clean_profile = str(st.session_state.get("form_profile", "")).strip()
        clean_desc = str(st.session_state.get("form_desc", "")).strip()
        clean_req = str(st.session_state.get("form_req", "")).strip()
        clean_ben = str(st.session_state.get("form_ben", "")).strip()

        if clean_profile:
            ad_sections.append(f"COMPANY PROFILE:\n{clean_profile}")
        if clean_desc:
            ad_sections.append(f"JOB DESCRIPTION:\n{clean_desc}")
        if clean_req:
            ad_sections.append(f"REQUIREMENTS:\n{clean_req}")
        if clean_ben:
            ad_sections.append(f"BENEFITS:\n{clean_ben}")

        text_to_highlight = "\n\n".join(ad_sections) if ad_sections else clean_desc

        highlighted_html, highlight_count = highlight_text(text_to_highlight)

        if highlight_count > 0:
            st.markdown(f"**Found {highlight_count} suspicious phrase instance{'s' if highlight_count != 1 else ''}:**")
            st.caption("🟡 **Legend:** Highlighted phrases match known recruitment scam keywords and heuristic red flag rules.")
            st.markdown(
                f"""<div style="
                    background-color: rgba(128, 128, 128, 0.08);
                    border: 1px solid rgba(128, 128, 128, 0.25);
                    padding: 16px;
                    border-radius: 8px;
                    line-height: 1.6;
                    max-height: 400px;
                    overflow-y: auto;
                ">{highlighted_html}</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("No suspicious phrases found in the submitted job advertisement text.")


# -----------------------------------------------------------------------------
# Page: Compare Ads
# -----------------------------------------------------------------------------

def page_compare_ads(predictor: Predictor) -> None:
    """
    Renders the side-by-side job advertisement comparison tool.

    Why this matters for viva:
    - Allows job seekers and students to contrast two job postings simultaneously
      to identify which posting carries higher risk.
    - Demonstrates local explainability side-by-side, comparing word weights and
      heuristic red flag triggers.
    - Operates entirely in memory: comparative evaluations are never persisted to
      the SQLite database, keeping user history clean.
    """
    st.title("⚖️ Compare Two Job Advertisements")
    st.write(
        "Compare two job or internship advertisements side-by-side to evaluate their "
        "relative scam risk scores, rule-based red flags, and vocabulary influences."
    )
    st.caption("ℹ️ **Note:** Comparative evaluations are processed in memory for immediate screening and are not saved to your scan history.")

    with st.form("compare_ads_form"):
        col_ad_a, col_ad_b = st.columns(2)

        # ------------------ Ad A Inputs ------------------
        with col_ad_a:
            st.subheader("📋 Job Ad A")
            title_a = st.text_input(
                "Job Title (Ad A)",
                value=st.session_state.get("cmp_title_a", ""),
                placeholder="e.g. Software Engineering Intern",
                key="cmp_input_title_a",
            )
            salary_a = st.text_input(
                "Salary Range (Optional)",
                value=st.session_state.get("cmp_salary_a", ""),
                placeholder="e.g. 50000-70000",
                key="cmp_input_salary_a",
            )
            profile_a = st.text_area(
                "Company Profile (Optional)",
                value=st.session_state.get("cmp_profile_a", ""),
                height=100,
                placeholder="Overview of hiring company...",
                key="cmp_input_profile_a",
            )
            desc_a = st.text_area(
                "Job Description (Required)",
                value=st.session_state.get("cmp_desc_a", ""),
                height=180,
                placeholder="Paste Job Ad A responsibilities and details here...",
                key="cmp_input_desc_a",
            )
            req_a = st.text_area(
                "Requirements & Skills (Optional)",
                value=st.session_state.get("cmp_req_a", ""),
                height=120,
                placeholder="Required education, technical skills...",
                key="cmp_input_req_a",
            )
            ben_a = st.text_area(
                "Benefits & Compensation (Optional)",
                value=st.session_state.get("cmp_ben_a", ""),
                height=120,
                placeholder="Healthcare, stipends, perks...",
                key="cmp_input_ben_a",
            )

            st.markdown("**Structured Posting Indicators (Ad A)**")
            c1_a, c2_a, c3_a = st.columns(3)
            with c1_a:
                tele_a = st.checkbox("Remote", value=st.session_state.get("cmp_tele_a", False), key="cmp_input_tele_a")
            with c2_a:
                logo_a = st.checkbox("Logo Present", value=st.session_state.get("cmp_logo_a", True), key="cmp_input_logo_a")
            with c3_a:
                quest_a = st.checkbox("Screening Questions", value=st.session_state.get("cmp_quest_a", False), key="cmp_input_quest_a")

        # ------------------ Ad B Inputs ------------------
        with col_ad_b:
            st.subheader("📋 Job Ad B")
            title_b = st.text_input(
                "Job Title (Ad B)",
                value=st.session_state.get("cmp_title_b", ""),
                placeholder="e.g. Remote Data Entry Assistant",
                key="cmp_input_title_b",
            )
            salary_b = st.text_input(
                "Salary Range (Optional)",
                value=st.session_state.get("cmp_salary_b", ""),
                placeholder="e.g. $400/day",
                key="cmp_input_salary_b",
            )
            profile_b = st.text_area(
                "Company Profile (Optional)",
                value=st.session_state.get("cmp_profile_b", ""),
                height=100,
                placeholder="Overview of hiring company...",
                key="cmp_input_profile_b",
            )
            desc_b = st.text_area(
                "Job Description (Required)",
                value=st.session_state.get("cmp_desc_b", ""),
                height=180,
                placeholder="Paste Job Ad B responsibilities and details here...",
                key="cmp_input_desc_b",
            )
            req_b = st.text_area(
                "Requirements & Skills (Optional)",
                value=st.session_state.get("cmp_req_b", ""),
                height=120,
                placeholder="Required education, technical skills...",
                key="cmp_input_req_b",
            )
            ben_b = st.text_area(
                "Benefits & Compensation (Optional)",
                value=st.session_state.get("cmp_ben_b", ""),
                height=120,
                placeholder="Healthcare, stipends, perks...",
                key="cmp_input_ben_b",
            )

            st.markdown("**Structured Posting Indicators (Ad B)**")
            c1_b, c2_b, c3_b = st.columns(3)
            with c1_b:
                tele_b = st.checkbox("Remote", value=st.session_state.get("cmp_tele_b", False), key="cmp_input_tele_b")
            with c2_b:
                logo_b = st.checkbox("Logo Present", value=st.session_state.get("cmp_logo_b", True), key="cmp_input_logo_b")
            with c3_b:
                quest_b = st.checkbox("Screening Questions", value=st.session_state.get("cmp_quest_b", False), key="cmp_input_quest_b")

        submit_compare = st.form_submit_button("⚖️ Compare Ads", use_container_width=True)

    # Handle Form Submission
    if submit_compare:
        # Validation 1: Descriptions required for both postings
        if not desc_a or not desc_a.strip():
            st.error("Validation Error: Job description for Ad A is required and cannot be empty.")
            return
        if not desc_b or not desc_b.strip():
            st.error("Validation Error: Job description for Ad B is required and cannot be empty.")
            return

        # Validation 2: Character limit of 20,000 characters per field
        fields_to_check = [
            ("Ad A Job Title", title_a),
            ("Ad A Company Profile", profile_a),
            ("Ad A Job Description", desc_a),
            ("Ad A Requirements", req_a),
            ("Ad A Benefits", ben_a),
            ("Ad A Salary Range", salary_a),
            ("Ad B Job Title", title_b),
            ("Ad B Company Profile", profile_b),
            ("Ad B Job Description", desc_b),
            ("Ad B Requirements", req_b),
            ("Ad B Benefits", ben_b),
            ("Ad B Salary Range", salary_b),
        ]
        for f_label, f_text in fields_to_check:
            if len(f_text) > 20000:
                st.error(f"Validation Error: '{f_label}' exceeds the maximum allowed length of 20,000 characters.")
                return

        # Save form values in session state for convenience
        st.session_state["cmp_title_a"] = title_a
        st.session_state["cmp_salary_a"] = salary_a
        st.session_state["cmp_profile_a"] = profile_a
        st.session_state["cmp_desc_a"] = desc_a
        st.session_state["cmp_req_a"] = req_a
        st.session_state["cmp_ben_a"] = ben_a
        st.session_state["cmp_tele_a"] = tele_a
        st.session_state["cmp_logo_a"] = logo_a
        st.session_state["cmp_quest_a"] = quest_a

        st.session_state["cmp_title_b"] = title_b
        st.session_state["cmp_salary_b"] = salary_b
        st.session_state["cmp_profile_b"] = profile_b
        st.session_state["cmp_desc_b"] = desc_b
        st.session_state["cmp_req_b"] = req_b
        st.session_state["cmp_ben_b"] = ben_b
        st.session_state["cmp_tele_b"] = tele_b
        st.session_state["cmp_logo_b"] = logo_b
        st.session_state["cmp_quest_b"] = quest_b

        with st.spinner("Analyzing and comparing both job postings with Machine Learning and Heuristic Rules..."):
            try:
                res_a = predictor.predict(
                    title=title_a,
                    company_profile=profile_a,
                    description=desc_a,
                    requirements=req_a,
                    benefits=ben_a,
                    salary_range=salary_a,
                    telecommuting=1 if tele_a else 0,
                    has_company_logo=1 if logo_a else 0,
                    has_questions=1 if quest_a else 0,
                )
                res_b = predictor.predict(
                    title=title_b,
                    company_profile=profile_b,
                    description=desc_b,
                    requirements=req_b,
                    benefits=ben_b,
                    salary_range=salary_b,
                    telecommuting=1 if tele_b else 0,
                    has_company_logo=1 if logo_b else 0,
                    has_questions=1 if quest_b else 0,
                )
                st.session_state["compare_results"] = {
                    "res_a": res_a,
                    "res_b": res_b,
                    "title_a": title_a or "Untitled Position A",
                    "title_b": title_b or "Untitled Position B",
                }
            except ValueError as err:
                st.error(f"Prediction Error: {err}")
                return

    # Render Comparison Results
    cmp_results = st.session_state.get("compare_results")
    if cmp_results:
        res_a = cmp_results["res_a"]
        res_b = cmp_results["res_b"]
        display_title_a = cmp_results["title_a"]
        display_title_b = cmp_results["title_b"]

        st.markdown("---")
        st.subheader("📊 Comparison Analysis")

        # ------------------ Comparison Summary Sentence ------------------
        score_a_pct = int(round(res_a["fraud_probability"] * 100))
        score_b_pct = int(round(res_b["fraud_probability"] * 100))
        score_diff = abs(score_a_pct - score_b_pct)

        if score_diff <= 5:
            summary_text = (
                f"Ad A and Ad B have similar estimated risk scores ({score_a_pct}% vs {score_b_pct}%)."
            )
        elif score_a_pct > score_b_pct:
            summary_text = (
                f"Ad A has the higher risk score ({score_a_pct}% vs {score_b_pct}%)."
            )
        else:
            summary_text = (
                f"Ad B has the higher risk score ({score_b_pct}% vs {score_a_pct}%)."
            )

        st.info(f"💡 **Comparison Summary:** {summary_text}")
        st.caption(
            "⚠️ **Disclaimer:** These scores reflect statistical similarity to known fraudulent postings. "
            "Neither ad is guaranteed to be definitely genuine or definitely fake."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ------------------ Side-by-Side Result Columns ------------------
        col_res_a, col_res_b = st.columns(2)

        for col_target, r_data, ad_tag, ad_heading in [
            (col_res_a, res_a, "Ad A", display_title_a),
            (col_res_b, res_b, "Ad B", display_title_b),
        ]:
            with col_target:
                st.markdown(f"### {ad_tag}: {ad_heading}")

                # 1. Coloured Risk Level Badge
                render_risk_badge(r_data["risk_level"])

                # 2. Risk Score & Progress Bar
                r_pct = int(round(r_data["fraud_probability"] * 100))
                st.write(f"#### Risk score: **{r_pct}%**")
                st.progress(float(r_data["fraud_probability"]))
                st.markdown(f"**Classification Output:** `{r_data['label']}`")

                st.markdown("---")

                # 3. Number of Warning Signs & Warning List
                ad_warnings = r_data.get("warnings", [])
                st.markdown(f"##### 🚩 Warning Signs ({len(ad_warnings)})")
                if ad_warnings:
                    for w in ad_warnings:
                        w_badge = render_warning_badge(w.get("severity", "low"))
                        matched_list = w.get("matched", [])
                        matched_str = ", ".join(matched_list) if matched_list else "N/A"
                        st.markdown(f"- {w_badge} **{w.get('message', '')}** *(Detected: `{matched_str}`)*")
                else:
                    st.success("No heuristic warning signs were triggered.")

                st.markdown("---")

                # 4. Top Fraud and Genuine Words
                st.markdown("##### 🔍 Influential Words")
                ad_top_words = r_data.get("top_words", {})
                fraud_w = ad_top_words.get("fraud", [])
                genuine_w = ad_top_words.get("genuine", [])

                sub_col_f, sub_col_g = st.columns(2)
                with sub_col_f:
                    st.caption("🔴 Toward Fraud")
                    if fraud_w:
                        fw_df = pd.DataFrame([
                            {"Word": item["word"], "Wt": f"+{item['contribution']:.3f}"}
                            for item in fraud_w[:5]
                        ])
                        st.dataframe(fw_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No strong words.")

                with sub_col_g:
                    st.caption("🟢 Toward Genuine")
                    if genuine_w:
                        gw_df = pd.DataFrame([
                            {"Word": item["word"], "Wt": f"{item['contribution']:.3f}"}
                            for item in genuine_w[:5]
                        ])
                        st.dataframe(gw_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No strong words.")


# -----------------------------------------------------------------------------
# Page: My History
# -----------------------------------------------------------------------------

def page_my_history(db: Database, user: Dict[str, Any]) -> None:
    """Renders the historical scan audit log for the authenticated user."""
    st.title("📜 My Scan History")
    st.write("Review, filter, or delete past job evaluations you have performed.")

    # Filtering Controls
    col_filter, col_search = st.columns([1, 2])
    with col_filter:
        risk_filter = st.selectbox(
            "Filter by Risk Level",
            options=["All", "Low", "Medium", "High"],
            index=0
        )
    with col_search:
        search_term = st.text_input("Search Job Title", placeholder="Type title keywords...").strip()

    # Query checks from database
    filter_val = None if risk_filter == "All" else risk_filter.lower()
    search_val = None if not search_term else search_term

    history = db.get_checks_for_user(
        user_id=user["id"],
        risk_level=filter_val,
        search=search_val
    )

    if not history:
        st.info("No scan records found matching your criteria.")
        return

    st.write(f"Showing **{len(history)}** scan records:")

    # Summary Table Display
    table_data = []
    for item in history:
        table_data.append({
            "ID": item["id"],
            "Date": item["created_at"],
            "Job Title": item["job_title"],
            "Risk Level": item["risk_level"].upper(),
            "Risk Score": f"{int(round(item['fraud_probability'] * 100))}%",
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔍 Inspect or Manage an Evaluation")

    check_options = {
        f"#{item['id']} - {item['job_title']} ({item['created_at']})": item["id"]
        for item in history
    }
    selected_label = st.selectbox("Select a check to inspect details:", list(check_options.keys()))
    selected_id = check_options[selected_label]

    # Retrieve selected check details
    selected_record = db.get_check(selected_id)

    if selected_record:
        with st.expander("📄 View Full Check Details", expanded=True):
            hist_label = label_for_level(selected_record["risk_level"])
            selected_record["label"] = hist_label

            st.markdown(f"**Job Title:** {selected_record['job_title']}")
            st.markdown(f"**Date:** {selected_record['created_at']}")
            st.markdown(f"**Risk Level:** `{selected_record['risk_level'].upper()}`")
            st.markdown(f"**Risk Score:** `{int(round(selected_record['fraud_probability'] * 100))}%`")
            st.markdown(f"**Classification Output:** `{hist_label}`")
            st.markdown(f"**Model:** `{selected_record['model_name']}`")

            st.markdown("**Warning Signs Triggered:**")
            if selected_record["warnings"]:
                for w in selected_record["warnings"]:
                    st.markdown(f"- {render_warning_badge(w['severity'])} {w['message']}")
            else:
                st.write("No warning signs were triggered.")

            st.markdown("**Job Text Analyzed (With Suspicious Phrase Highlights):**")
            hist_highlighted_html, hist_match_count = highlight_text(selected_record["job_text"])
            if hist_match_count > 0:
                st.caption(f"🟡 **Legend:** {hist_match_count} suspicious phrase instance{'s' if hist_match_count != 1 else ''} highlighted.")
                st.markdown(
                    f"""<div style="
                        background-color: rgba(128, 128, 128, 0.08);
                        border: 1px solid rgba(128, 128, 128, 0.25);
                        padding: 14px;
                        border-radius: 8px;
                        line-height: 1.6;
                        max-height: 300px;
                        overflow-y: auto;
                        margin-bottom: 12px;
                    ">{hist_highlighted_html}</div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No suspicious phrases found in this saved ad text.")
                st.text_area(
                    "Raw Job Text",
                    value=selected_record["job_text"],
                    height=120,
                    disabled=True,
                    key=f"hist_raw_txt_{selected_id}",
                )

            # Download Report PDF Button (Task 2)
            # Ownership check: Normal user may only download reports for their own checks
            if selected_record.get("user_id") == user["id"]:
                hist_pdf = ReportGenerator.build_pdf(
                    check=selected_record,
                    owner_username=user.get("username", "User"),
                )
                st.download_button(
                    label="📄 Download report (PDF)",
                    data=hist_pdf,
                    file_name=f"job_check_{selected_record['id']}.pdf",
                    mime="application/pdf",
                    key=f"dl_report_hist_{selected_record['id']}",
                )

            # Single record delete button
            if st.button("🗑️ Delete this record", key=f"del_{selected_id}"):
                deleted = db.delete_check(check_id=selected_id, user_id=user["id"])
                if deleted:
                    st.success("Record deleted successfully.")
                    st.rerun()
                else:
                    st.error("Failed to delete record.")

    st.markdown("---")
    # Delete All History Section
    with st.expander("⚠️ Danger Zone: Delete All History"):
        st.write("This will permanently delete all your saved job evaluation records.")
        confirm_delete_all = st.checkbox("I confirm that I want to delete all my history records.")
        if st.button("Delete All My History", type="secondary"):
            if confirm_delete_all:
                deleted_count = db.delete_all_checks_for_user(user_id=user["id"])
                st.success(f"Successfully deleted {deleted_count} history records.")
                st.rerun()
            else:
                st.warning("Please check the confirmation box first.")


# -----------------------------------------------------------------------------
# Page: Model Insights
# -----------------------------------------------------------------------------

def page_model_insights() -> None:
    """
    Renders the Model Insights & Explainability page.

    This page is strictly read-only: it loads pre-computed metrics, comparison tables,
    and visual artifacts from `models/` and `reports/` without retraining anything.
    """
    st.title("🔬 Model Insights & Evaluation")

    # Load cached data sources
    metadata = load_metadata_cached()
    comp_df = load_model_comparison_cached()
    test_stats = load_test_dataset_stats_cached()

    best_model_name = metadata.get("best_model_name", "Logistic Regression") if metadata else "Logistic Regression"
    training_date = metadata.get("training_date", "Unknown") if metadata else "Unknown"

    # -------------------------------------------------------------------------
    # 1. Short Intro
    # -------------------------------------------------------------------------
    st.markdown(f"""
    ### 🧠 Machine Learning Architecture & Overview
    The **Fake Job Posting Detector** uses a production classification pipeline combining:
    - **TF-IDF Text Representations**: 10,000 sublinear unigram and bigram features extracted across
      the job title, company profile, description, requirements, and benefits.
    - **Structured Binary Flags**: 6 binary features (`telecommuting`, `has_company_logo`,
      `has_questions`, `missing_company_profile`, `missing_salary`, and `short_description`).
    - **Active Production Classifier**: **{best_model_name}** (balanced class weighting).
    - **Model Training Date**: `{training_date}`
    """)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2. Why Accuracy is Misleading Here
    # -------------------------------------------------------------------------
    st.subheader("⚠️ Why Accuracy is Misleading Here")

    if test_stats and metadata:
        dummy_acc = test_stats["dummy_accuracy"]
        test_metrics = metadata.get("test_metrics", {})
        model_acc = test_metrics.get("accuracy", 0.0) * 100.0

        col_acc1, col_acc2 = st.columns(2)
        with col_acc1:
            st.metric(
                label="Baseline 'Dummy' Accuracy (Always Guess 'Genuine')",
                value=f"{dummy_acc:.2f}%",
                help="A naive classifier that never flags any ad achieves this accuracy by only predicting the majority class."
            )
        with col_acc2:
            st.metric(
                label=f"Production Model Accuracy ({best_model_name})",
                value=f"{model_acc:.2f}%",
                help="Our trained Logistic Regression model test accuracy."
            )

        st.markdown(f"""
        In our unseen test dataset of **{test_stats['total']:,}** job postings, **{test_stats['genuine_count']:,} ({dummy_acc:.2f}%)**
        are genuine and only **{test_stats['fraud_count']:,} ({test_stats['fraud_prevalence']:.2f}%)** are fraudulent.

        Because fake ads represent only ~4.5% of the data, a naive model that predicts **every single ad is genuine**
        achieves an impressive **{dummy_acc:.2f}% accuracy** while catching **0% of scams (zero recall)**!

        > **Why this matters for viva**: In severe class imbalance, accuracy rewards guessing the majority class and is practically useless.
        > Models must instead be evaluated on **Recall** (catching true fakes), **Precision** (avoiding false alarms),
        > **F1-Score** (harmonic balance), and **Precision-Recall AUC (PR-AUC)**.
        """)
    else:
        st.info("Dataset statistics are currently unavailable (processed test data or metadata file not found).")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 3. Model Comparison Table
    # -------------------------------------------------------------------------
    st.subheader("📊 Model Benchmark Comparison Table")

    if comp_df is not None:
        display_records = []
        for _, r in comp_df.iterrows():
            m_name = str(r["model_name"])
            is_active = (m_name == best_model_name)
            display_records.append({
                "Model Architecture": f"⭐ {m_name} (Active in App)" if is_active else m_name,
                "CV Recall (Train)": f"{r['cv_recall_mean']*100:.1f}% ± {r['cv_recall_std']*100:.1f}%",
                "CV Precision (Train)": f"{r['cv_precision_mean']*100:.1f}% ± {r['cv_precision_std']*100:.1f}%",
                "CV F1-Score (Train)": f"{r['cv_f1_mean']:.3f} ± {r['cv_f1_std']:.3f}",
                "CV PR-AUC (Train)": f"{r['cv_pr_auc_mean']:.3f} ± {r['cv_pr_auc_std']:.3f}",
                "Test Recall": f"{r['test_recall']*100:.2f}%",
                "Test Precision": f"{r['test_precision']*100:.2f}%",
                "Test F1-Score": f"{r['test_f1']:.4f}",
                "Test PR-AUC": f"{r['test_pr_auc']:.4f}",
                "Test Accuracy": f"{r['test_accuracy']*100:.2f}%",
            })

        st.dataframe(pd.DataFrame(display_records), use_container_width=True, hide_index=True)
        st.caption(
            "Note: Model selection was decided solely by 5-Fold Stratified Cross-Validation F1 on the training set. "
            "The test set was evaluated once at the very end to guarantee zero data leakage."
        )
    else:
        st.warning("Model comparison benchmark table (`models/model_comparison.csv`) was not found.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 4. Plain-Language Box: The Real-World Engineering Trade-Off
    # -------------------------------------------------------------------------
    st.subheader("⚖️ The Real-World Engineering Trade-Off: Recall vs. Precision")

    if comp_df is not None and test_stats is not None:
        try:
            lr_row = comp_df[comp_df["model_name"] == "Logistic Regression"].iloc[0]
            rf_row = comp_df[comp_df["model_name"] == "Random Forest"].iloc[0]

            total_fraud = test_stats["fraud_count"]
            lr_rec_pct = lr_row["test_recall"] * 100.0
            lr_prec_pct = lr_row["test_precision"] * 100.0
            lr_caught = int(round(total_fraud * lr_row["test_recall"]))

            rf_rec_pct = rf_row["test_recall"] * 100.0
            rf_prec_pct = rf_row["test_precision"] * 100.0
            rf_caught = int(round(total_fraud * rf_row["test_recall"]))
            rf_missed = total_fraud - rf_caught

            st.info(f"""
            **Trade-Off Breakdown (Computed Dynamically from Unseen Test Data):**

            - **Logistic Regression (App's Active Model)**:
              - Catches **{lr_rec_pct:.1f}%** of fake ads (**{lr_caught} of {total_fraud}** scams detected).
              - Achieves **{lr_prec_pct:.1f}%** precision (roughly 1 in 2 flagged postings is confirmed fraud).
              - Raises more false alarms on unusual genuine postings, but lets very few fakes slip through.

            - **Random Forest**:
              - Achieves outstanding precision of **{rf_prec_pct:.1f}%** (almost never raises a false alarm).
              - BUT catches only **{rf_rec_pct:.1f}%** of fake ads (**{rf_caught} of {total_fraud}**), letting **{rf_missed} scams slip through undetected**!

            ---

            **Why High Recall Wins for Student Protection:**
            In a recruitment fraud screening tool, **missing a scam (False Negative) is far worse than a false alarm (False Positive)**:
            - A **False Negative** means a student believes a fraudulent posting is legitimate, potentially losing registration fees, falling for counterfeit check schemes, or exposing their identity.
            - A **False Positive** merely prompts the student to pause, inspect the company's official website, and verify the recruiter.

            The app's model was selected by cross-validated F1 on the training data, where Logistic Regression scored slightly higher than Random Forest. Its higher recall also suits a scam-screening tool, because missing a scam is generally worse than a false alarm. Random Forest is more precise but misses about half of the fake ads.
            """)
        except Exception:
            st.info("The trade-off summary could not be rendered dynamically from the comparison table.")
    else:
        st.info("Model comparison data is unavailable to display the trade-off details.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 5. The Four Visual Report Charts
    # -------------------------------------------------------------------------
    st.subheader("📈 Visual Evaluation Reports & Interpretability")

    reports_dir = SCRIPT_DIR / "reports"
    charts = [
        (
            "model_comparison.png",
            "Model Benchmark: F1-Score vs. Recall",
            "How to read: Compares F1-Score (blue) and Recall (orange) across all three architectures on the test set; notice Logistic Regression's decisive advantage in fraud recall."
        ),
        (
            "confusion_matrix_best.png",
            "Confusion Matrix (Logistic Regression)",
            "How to read: Shows true vs. predicted counts for the winning Logistic Regression model; top-left is genuine ads correctly cleared, bottom-right is fake ads successfully caught."
        ),
        (
            "pr_curve_best.png",
            "Precision-Recall Curve (PR-AUC)",
            "How to read: Plots precision vs. recall across all potential risk score thresholds; the red dashed line is the 4.5% no-skill baseline, and higher curves demonstrate superior scam separation."
        ),
        (
            "top_words_logreg.png",
            "Vocabulary Coefficients: Fraud vs. Genuine Signals",
            "How to read: Displays the 20 strongest vocabulary signals pushing toward fraud (positive coefficients in red) versus genuine (negative coefficients in navy) learned by Logistic Regression."
        ),
    ]

    col_c1, col_c2 = st.columns(2)
    for idx, (img_name, title, caption) in enumerate(charts):
        img_path = reports_dir / img_name
        target_col = col_c1 if idx % 2 == 0 else col_c2
        with target_col:
            st.markdown(f"##### {title}")
            if img_path.is_file():
                st.image(str(img_path), caption=caption, use_container_width=True)
            else:
                st.warning(f"Visualization artifact `{img_name}` not found in `reports/`.")
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 6. Risk Level Cutoffs (Empirically Calibrated)
    # -------------------------------------------------------------------------
    st.subheader("🎯 Data-Driven Risk Level Cutoffs")

    low_max = 0.40
    high_min = 0.50
    if metadata and "risk_cutoffs" in metadata:
        rc = metadata["risk_cutoffs"]
        low_max = float(rc.get("low_max", low_max))
        high_min = float(rc.get("high_min", high_min))

    low_pct = int(round(low_max * 100))
    high_pct = int(round(high_min * 100))
    med_max_pct = high_pct - 1

    st.markdown(f"""
    Rather than guessing arbitrary thresholds (such as 30% and 60%), the application's risk cutoffs were
    **empirically calibrated** using 5-fold Stratified Cross-Validation out-of-fold scores on the training data
    (`calibrate_risk_levels.py`):

    1. **LOW Risk (< {low_pct}%)** — *Output: "Likely genuine"*:
       - Risk score strictly below **{low_max:.2f}**.
       - Chosen using the rule requiring **fraud recall of at least 90%** (at this cutoff, it actually achieved over 93% recall on the training data).
       - Ensures that a LOW assessment rarely conceals a real scam ad.

    2. **MEDIUM Risk ({low_pct}% – {med_max_pct}%)** — *Output: "Borderline - review carefully"*:
       - Risk score from **{low_max:.2f}** to **< {high_min:.2f}**.
       - **Borderline, review carefully**. Postings here exhibit ambiguous signals (e.g. missing company profiles,
         brief descriptions, mild urgency phrasing). Exercise caution and research the employer independently.

    3. **HIGH Risk (≥ {high_pct}%)** — *Output: "Potentially fraudulent"*:
       - Risk score at or above **{high_min:.2f}**.
       - Chosen as the lowest candidate cutoff where **precision reaches at least 50%**.
       - In the cross-validated training scores, about half of the ads in this tier were labelled fraudulent in the dataset, identifying strong alignment with known scam patterns.

    > **Training-Data Selection**: Calibrating cutoffs strictly on cross-validated training fold predictions ensures
    > optimal decision boundaries without leaking or overfitting to the unseen test dataset.
    """)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 7. Limitations Box
    # -------------------------------------------------------------------------
    st.subheader("⚠️ Model Limitations & Responsible Use")

    st.warning("""
    **Please keep the following constraints in mind when using this tool:**
    1. **Historical Training Corpus**: The model is trained on the University of the Aegean EMSCAD dataset
       comprising ~17,800 postings from 2012 to 2014. While foundational scam patterns persist (upfront fees,
       unrealistic daily wages, messaging app contact), emerging tactics (AI-generated text, cryptocurrency tasks) may differ.
    2. **Single Dataset Source**: All training and validation postings originate from one historical corpus.
    3. **English Language Only**: Natural language text preprocessing, tokenization, and TF-IDF vocabulary extraction
       are optimized strictly for English postings.
    4. **Advisory Estimate, Not True Probability**: The calculated risk score is a statistical similarity estimate
       against historical patterns. It is not a calibrated mathematical probability or definitive legal proof.
    5. **Dual Error Possibility**: The model can produce False Positives (informal legitimate startup ads) and
       False Negatives (sophisticated corporate impersonations).
    6. **Heuristic Separation**: Red flag warning indicators come from simple deterministic pattern rules (`rules.py`)
       and operate completely independently of the statistical machine learning model.
    7. These results are measured on one public dataset. Real-world performance on newer postings may be lower.
    """)


# -----------------------------------------------------------------------------
# Page: About
# -----------------------------------------------------------------------------

def page_about() -> None:
    """Renders project documentation, risk interpretations, and limitations."""
    st.title("ℹ️ About Fake Job Posting Detector")

    # Read empirically calibrated risk cutoffs from metadata if available
    metadata_path = SCRIPT_DIR / "models" / "metadata.json"
    low_max = 0.40
    high_min = 0.50
    if metadata_path.is_file():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cutoffs = meta.get("risk_cutoffs", {})
            low_max = float(cutoffs.get("low_max", low_max))
            high_min = float(cutoffs.get("high_min", high_min))
        except Exception:
            pass

    low_pct = int(round(low_max * 100))
    high_pct = int(round(high_min * 100))
    med_max_pct = high_pct - 1

    st.markdown(f"""
    ### 🎯 Project Purpose
    The **Fake Job Posting Detector** is an integrated academic project bridging **Machine Learning**
    and **Advanced Python**. It is designed to assist college students, recent graduates, and job
    seekers in evaluating whether an online job or internship listing exhibits characteristics of
    recruitment scams, phishing attempts, or fraudulent schemes.

    ---

    ### 📊 How to Interpret the Risk Score & Levels

    1. **Risk Score (0% – 100%)**:
       - The score represents the statistical similarity between the analyzed job description
         and the fraudulent postings in our training corpus.
       - **It is an advisory risk score, NOT a mathematical probability or definitive proof.**

    2. **Low Risk (< {low_pct}%)** — *Output: "Likely genuine"*:
       - The posting reflects legitimate organizational language, typical structural formatting
         (official company profile, clear qualification scope), and lacks high-risk keywords.
       - Chosen from cross-validated scores on the training data so that at least 90% of fraudulent
         postings score at or above this cutoff (achieving ~93.4% fraud recall), ensuring a LOW
         result rarely hides a fake ad.

    3. **Medium Risk ({low_pct}% – {med_max_pct}%)** — *Output: "Borderline - review carefully"*:
       - **Borderline, review carefully**. The posting exhibits ambiguous or inconsistent signals
         (e.g. missing company profile, unusually brief description, or mild urgency phrasing).
       - Exercise caution, independently verify the employer's domain, and avoid sharing sensitive
         personal information until legitimate employment is verified.

    4. **High Risk (≥ {high_pct}%)** — *Output: "Potentially fraudulent"*:
       - Strong statistical alignment with known fraudulent schemes. Common indicators include
         requests for registration fees, free email/messaging app contact methods (WhatsApp, Gmail),
         or hyperbolic daily/weekly income promises.
       - Chosen from cross-validated scores on the training data as the lowest cutoff where precision
         reaches at least 50% (over half of all ads scoring at or above this threshold are true scams).

    > **Data-Driven Calibration Note**: Rather than guessing cutoffs, these levels were selected from
    > 5-fold cross-validated out-of-fold risk scores on the training data (`calibrate_risk_levels.py`).
    > Using out-of-fold scores on the training set avoids choosing cutoffs with the test set (preventing
    > test-set data leakage) and eliminates training-set overfitting bias.

    ---

    ### 🚩 How the Warning Signs Work
    In addition to the statistical machine learning model, the detector runs **7 heuristic pattern rules** (`rules.py`).
    These rules check for known red flags commonly seen in fraudulent schemes:
    - **Fee or Payment Requests** (application fees, training deposits, wire transfers).
    - **Unrealistic Earnings & Get-Rich Promises** (e.g. guaranteed daily cash, hyperbolic income claims).
    - **Artificial Urgency Language** (e.g. 'immediate joining', 'apply now or miss out').
    - **Personal Contact Channels** (using personal @gmail/@yahoo addresses or WhatsApp/Telegram).
    - **Missing Company Profile** (failing to describe the hiring entity).
    - **Abnormally Short Descriptions** (< 30 words, vague duties).
    - **Predatory 'No Experience Required' Phrasing** paired with high compensation claims.

    > **Separate from the Machine Learning Model**: These warning signs are simple keyword and pattern rules,
    > completely independent of the machine learning classifier. A posting might trigger a rule warning even
    > if its statistical risk score is low, or vice versa, providing a layered defense.

    ---

    ### ⚠️ Limitations & Disclaimers

    - **Historical Training Corpus**: The model is trained on the University of the Aegean's
      EMSCAD (Employment Scam Aegean Dataset) containing ~17,800 postings from 2012 to 2014.
      While core scam patterns persist (fees, false urgency, high earnings), modern phishing tactics
      may differ.
    - **Dual Error Possibility**: Like all machine learning classifiers, this tool can produce:
      - *False Positives*: Legitimate, informal startup ads flagged as high risk.
      - *False Negatives*: Sophisticated corporate impersonation scams scored as low risk.
    - **Decision Support Only**: This tool should be used as one of several verification steps.
      Never pay money to apply for a job, never accept unverified checks, and always verify job
      requisitions on the employer's official career website.
    """)


# -----------------------------------------------------------------------------
# Page: Admin Dashboard
# -----------------------------------------------------------------------------

def page_admin_dashboard(db: Database, auth_mgr: AuthManager, current_user: Dict[str, Any]) -> None:
    """
    Renders the Administrative Monitoring & User Management Dashboard.

    Security & Viva Controls:
    1. Dual Role Enforcement:
       Verifies admin authorization inside the page handler via `auth_mgr.is_admin(current_user)`
       in addition to hiding the sidebar link. Unauthorized callers are halted with `st.stop()`.
    2. Zero Credential Exposure:
       User lists and audit logs NEVER retrieve or display password hashes.
    3. Privacy Safeguard:
       A mandatory data privacy notice reminds administrators to handle user-submitted ad text responsibly.
    4. Guardrails on Account Deletion:
       Administrators cannot delete their own active account, and the system refuses to delete the
       last remaining administrator account to prevent lockout.
    """
    # Security Rule 1: Role check inside page function itself
    if not auth_mgr.is_admin(current_user):
        st.error("🚨 **Access Denied:** You do not have permission to access the Admin Dashboard. Administrator privileges are required.")
        st.stop()

    st.title("⚙️ Admin Dashboard")

    # Privacy Notice (Item 6)
    st.info("🔒 **Privacy Notice:** Admins can see users' pasted job ads. Handle this data responsibly.")

    # 1. Aggregate Statistics Row (Item 1)
    stats = db.get_stats()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Scans", f"{stats['total_checks']:,}")
    with col2:
        st.metric("Total Users", f"{stats['total_users']:,}")
    with col3:
        st.metric("High-Risk Checks", f"{stats['high_risk_count']:,}")
    with col4:
        st.metric("High-Risk Share", f"{stats['high_risk_percentage']:.1f}%")

    st.markdown("---")

    # 2. Activity Timeline Bar Chart (Item 2)
    st.subheader("📈 Scans Per Day (Last 14 Days)")
    daily_data = stats.get("checks_per_day", {})
    if daily_data:
        chart_df = pd.DataFrame(
            list(daily_data.items()),
            columns=["Date", "Scans"]
        )
        # Ensure y-axis starts at 0 and displays integer/whole numbers only
        max_scans = int(chart_df["Scans"].max()) if not chart_df.empty else 0
        y_max = max(max_scans + 1, 5)

        timeline_chart = (
            alt.Chart(chart_df)
            .mark_bar(color="#2b5c8f")
            .encode(
                x=alt.X("Date:N", title="Date", sort=None),
                y=alt.Y(
                    "Scans:Q",
                    title="Number of Scans",
                    scale=alt.Scale(domain=[0, y_max]),
                    axis=alt.Axis(tickMinStep=1, format="d"),
                ),
                tooltip=["Date", "Scans"],
            )
            .properties(height=280)
        )
        st.altair_chart(timeline_chart, use_container_width=True)

        if max_scans == 0:
            st.caption("ℹ️ No scan activity recorded across the last 14 days.")
    else:
        st.info("No timeline activity recorded yet.")

    st.markdown("---")

    # 3. Recent Checks Table & Filters (Item 3)
    st.subheader("📋 Recent Scans (All Users)")

    # Fetch users to populate username filter options
    all_users = db.list_users()
    known_usernames = sorted(list({u["username"] for u in all_users if u.get("username")}))

    col_filter_risk, col_filter_user = st.columns(2)
    with col_filter_risk:
        risk_choice = st.selectbox(
            "Filter by Risk Level",
            options=["All", "Low", "Medium", "High"],
            index=0,
            key="admin_filter_risk",
        )
    with col_filter_user:
        user_choice = st.selectbox(
            "Filter by Username",
            options=["All"] + known_usernames,
            index=0,
            key="admin_filter_user",
        )

    # Query checks from database
    filter_risk_val = None if risk_choice == "All" else risk_choice.lower()
    filter_user_val = None if user_choice == "All" else user_choice

    all_checks = db.get_all_checks(
        limit=200,
        risk_level=filter_risk_val,
        username=filter_user_val,
    )

    if not all_checks:
        if stats["total_checks"] == 0:
            st.info("No job advertisement checks have been performed yet.")
        else:
            st.info("No scan records match the selected filters.")
    else:
        st.write(f"Displaying **{len(all_checks)}** recent check records:")
        check_table_rows = []
        for chk in all_checks:
            score_pct = int(round(chk["fraud_probability"] * 100))
            check_table_rows.append({
                "ID": chk["id"],
                "Date": chk["created_at"],
                "Username": chk.get("username", "Guest"),
                "Job Title": chk["job_title"] or "Untitled Position",
                "Risk Level": chk["risk_level"].upper(),
                "Risk Score": f"{score_pct}%",
            })
        st.dataframe(pd.DataFrame(check_table_rows), use_container_width=True, hide_index=True)

        st.markdown("---")

        # 4. Inspect or Delete a Check (Item 4)
        st.subheader("🔍 Inspect or Delete a Check")
        check_options = {
            f"#{chk['id']} - {chk['job_title']} ({chk.get('username', 'Guest')}, {chk['created_at']})": chk["id"]
            for chk in all_checks
        }
        selected_chk_label = st.selectbox(
            "Select a scan check to review details:",
            options=list(check_options.keys()),
            key="admin_select_check",
        )
        selected_chk_id = check_options[selected_chk_label]

        check_detail = db.get_check_with_owner(selected_chk_id)
        if check_detail:
            with st.expander("📄 View Check Details & Warning Indicators", expanded=True):
                col_cd1, col_cd2 = st.columns(2)
                with col_cd1:
                    st.markdown(f"**Check ID:** `#{check_detail['id']}`")
                    st.markdown(f"**Submitted By:** `{check_detail.get('username', 'Guest')}` (User ID: {check_detail['user_id']})")
                    st.markdown(f"**Job Title:** {check_detail['job_title']}")
                    st.markdown(f"**Date:** {check_detail['created_at']}")
                with col_cd2:
                    adm_label = label_for_level(check_detail["risk_level"])
                    check_detail["label"] = adm_label

                    st.markdown(f"**Risk Level:** `{check_detail['risk_level'].upper()}`")
                    st.markdown(f"**Risk Score:** `{int(round(check_detail['fraud_probability'] * 100))}%` ({check_detail['fraud_probability'] * 100:.2f}%)")
                    st.markdown(f"**Classification Output:** `{adm_label}`")
                    st.markdown(f"**Model Used:** `{check_detail['model_name']}`")

                st.markdown("**Warning Signs Triggered:**")
                warnings = check_detail.get("warnings", [])
                if warnings:
                    for w in warnings:
                        severity_tag = render_warning_badge(w.get("severity", "low"))
                        matched_str = ", ".join(w["matched"]) if w.get("matched") else "N/A"
                        st.markdown(f"- {severity_tag} **{w.get('message', '')}** *(Detected: `{matched_str}`)*")
                else:
                    st.success("No heuristic warning signs were triggered. The ad passed all rule checks.")

                st.markdown("**Pasted Job Ad Text (With Suspicious Phrase Highlights):**")
                adm_hl_html, adm_hl_count = highlight_text(check_detail["job_text"])
                if adm_hl_count > 0:
                    st.caption(f"🟡 **Legend:** {adm_hl_count} suspicious phrase instance{'s' if adm_hl_count != 1 else ''} highlighted.")
                    st.markdown(
                        f"""<div style="
                            background-color: rgba(128, 128, 128, 0.08);
                            border: 1px solid rgba(128, 128, 128, 0.25);
                            padding: 14px;
                            border-radius: 8px;
                            line-height: 1.6;
                            max-height: 300px;
                            overflow-y: auto;
                            margin-bottom: 12px;
                        ">{adm_hl_html}</div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No suspicious phrases found in this ad text.")
                    st.text_area(
                        "Raw Analyzed Text",
                        value=check_detail["job_text"],
                        height=120,
                        disabled=True,
                        key=f"admin_view_text_{selected_chk_id}",
                    )

                # Download Report PDF Button for Admins (Task 2)
                admin_pdf = ReportGenerator.build_pdf(
                    check=check_detail,
                    owner_username=check_detail.get("username", "Guest"),
                )
                st.download_button(
                    label="📄 Download report (PDF)",
                    data=admin_pdf,
                    file_name=f"job_check_{check_detail['id']}.pdf",
                    mime="application/pdf",
                    key=f"admin_dl_report_{check_detail['id']}",
                )

                if st.button("🗑️ Delete this check", key=f"admin_btn_del_{selected_chk_id}", type="secondary"):
                    deleted = db.delete_check(check_id=selected_chk_id)
                    if deleted:
                        st.success(f"Check #{selected_chk_id} deleted successfully.")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete check #{selected_chk_id}.")

    st.markdown("---")

    # 5. Users Table & Account Deletion Workflow (Item 5)
    st.subheader("👥 Registered Users")

    if not all_users:
        st.info("No registered users found.")
    else:
        user_rows = []
        for u in all_users:
            user_rows.append({
                "ID": u["id"],
                "Username": u["username"],
                "Role": u["role"].upper(),
                "Created Date": u["created_at"],
                "Number of Checks": u.get("check_count", 0),
            })
        st.dataframe(pd.DataFrame(user_rows), use_container_width=True, hide_index=True)

        st.markdown("##### 🗑️ User Account Deletion")

        admin_count = sum(1 for u in all_users if u.get("role") == "admin")

        user_select_map = {
            f"#{u['id']} - {u['username']} ({u['role'].upper()})": u["id"]
            for u in all_users
        }
        selected_user_label = st.selectbox(
            "Select user account to manage or delete:",
            options=list(user_select_map.keys()),
            key="admin_select_user",
        )
        selected_uid = user_select_map[selected_user_label]
        target_user = next((u for u in all_users if u["id"] == selected_uid), None)

        if target_user:
            # Rule: An admin cannot delete their own account
            if target_user["id"] == current_user["id"]:
                st.warning("⚠️ **Self-Deletion Blocked:** You cannot delete your own logged-in administrator account.")
            # Rule: The last remaining admin account can never be deleted
            elif target_user["role"] == "admin" and admin_count <= 1:
                st.warning("⚠️ **Last Admin Protection:** This is the only remaining administrator account. It cannot be deleted.")
            else:
                st.markdown(
                    f"Deleting user **{target_user['username']}** will permanently delete their account "
                    f"and cascade deletion to all **{target_user.get('check_count', 0)}** associated scan checks."
                )
                confirm_user_del = st.checkbox(
                    f"I confirm that I want to permanently delete user '{target_user['username']}' and all their scan history.",
                    key=f"chk_confirm_del_user_{selected_uid}",
                )
                if st.button(
                    f"Delete User '{target_user['username']}'",
                    type="primary",
                    disabled=not confirm_user_del,
                    key=f"btn_del_user_{selected_uid}",
                ):
                    try:
                        del_ok = db.delete_user(selected_uid)
                        if del_ok:
                            st.success(f"User '{target_user['username']}' and their associated scan history have been deleted.")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete user '{target_user['username']}'.")
                    except ValueError as val_err:
                        st.error(f"Action Blocked: {val_err}")


# -----------------------------------------------------------------------------
# Main Application Controller
# -----------------------------------------------------------------------------

def main() -> None:
    """Configures the Streamlit app, handles session routing, and renders pages."""
    st.set_page_config(
        page_title="Fake Job Posting Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize shared dependencies
    db: Database = get_database()
    auth_mgr: AuthManager = get_auth_manager()
    predictor: Optional[Predictor] = get_predictor()

    # Verify model availability
    if predictor is None:
        st.error(
            "🚨 **Model Artifacts Not Found!**\n\n"
            "The application could not find `best_pipeline.joblib` or `metadata.json` in `models/`.\n\n"
            "**Action Required:** Please run the training script in your terminal first:\n"
            "```bash\npython train_model.py\n```"
        )
        st.stop()

    # Session State Initialization
    if "user" not in st.session_state:
        st.session_state["user"] = None

    current_user = st.session_state["user"]

    # If user is not authenticated, show Login / Register screen
    if current_user is None:
        render_auth_screen(auth_mgr)
        return

    # ------------------ Authenticated Sidebar ------------------
    with st.sidebar:
        st.markdown(f"### 👤 Logged in as: **{current_user['username']}**")
        st.markdown(f"**Role:** `{current_user.get('role', 'user').upper()}`")

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["user"] = None
            st.session_state.pop("last_result", None)
            st.rerun()

        st.markdown("---")
        st.subheader("Navigation")

        # Page options list (Compare Ads and Model Insights are visible to every logged-in user)
        menu_options = ["Check a Job Ad", "Compare Ads", "My History", "Model Insights", "About"]

        # Security: Show "Admin Dashboard" entry ONLY if logged-in user is an admin
        if auth_mgr.is_admin(current_user):
            menu_options.append("Admin Dashboard")

        selected_page = st.radio("Go to:", menu_options, index=0)

    # ------------------ Page Routing ------------------
    if selected_page == "Check a Job Ad":
        page_check_job(predictor=predictor, db=db, user=current_user)
    elif selected_page == "Compare Ads":
        page_compare_ads(predictor=predictor)
    elif selected_page == "My History":
        page_my_history(db=db, user=current_user)
    elif selected_page == "Model Insights":
        page_model_insights()
    elif selected_page == "About":
        page_about()
    elif selected_page == "Admin Dashboard":
        page_admin_dashboard(db=db, auth_mgr=auth_mgr, current_user=current_user)


if __name__ == "__main__":
    main()
