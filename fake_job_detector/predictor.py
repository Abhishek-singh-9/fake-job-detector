"""
Prediction & Explainability Pipeline for Fake Job Posting Detector.

This module provides the Predictor class, which combines machine learning inference
with rule-based heuristics and local feature explainability.

Key Architecture & Viva Points:
- Feature Consistency:
  Reuses `clean_text_content` directly from `prepare_data.py`. This guarantees that text
  is tokenized, stripped of HTML tags, lowercased, and whitespace-collapsed identically
  between training and live inference.
- Explainability via Logistic Regression Contributions:
  For any job description, the contribution of each word is computed as:
      Contribution = TF-IDF(word) * Coefficient(word)
  Words with positive contributions push the score toward FRAUD; words with negative
  contributions push the score toward GENUINE.
- Advisory Estimate, Never Proof:
  The calculated probability is a statistical estimate reflecting patterns observed in
  historical data. It must be presented as decision support, never definitive legal proof.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import joblib
import pandas as pd

# Add project root directory to sys.path so sibling imports work seamlessly
SCRIPT_DIR: Path = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the exact same text cleaning logic as used during model training
from prepare_data import clean_text_content
from rules import RuleChecker

# Fixed random seed
RANDOM_STATE: int = 42

# Fallback Risk Level Thresholds (used only if models/metadata.json lacks 'risk_cutoffs')
DEFAULT_RISK_LOW_MAX: float = 0.30       # Fallback: strictly below 0.30 -> Low Risk
DEFAULT_RISK_HIGH_MIN: float = 0.60      # Fallback: at or above 0.60 -> High Risk
RISK_LOW_MAX: float = DEFAULT_RISK_LOW_MAX
RISK_MEDIUM_MAX: float = DEFAULT_RISK_HIGH_MIN


class Predictor:
    """
    Main prediction engine combining trained ML models, rule checks, and explainability.
    """

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        """
        Loads the trained model artifacts, vectorizers, and metadata.

        Why this matters:
        Using relative paths from Path(__file__) ensures model artifacts load reliably
        regardless of the working directory from which Python or Streamlit is executed.

        Args:
            models_dir (Optional[Path]): Directory containing .joblib and .json files.
        """
        # Resolve models directory relative to this script
        base_dir: Path = models_dir if models_dir is not None else (SCRIPT_DIR / "models")

        best_pipeline_path = base_dir / "best_pipeline.joblib"
        logreg_pipeline_path = base_dir / "logreg_pipeline.joblib"
        metadata_path = base_dir / "metadata.json"

        if not best_pipeline_path.is_file() or not logreg_pipeline_path.is_file():
            raise FileNotFoundError(
                f"Model pipeline artifacts not found in {base_dir}. "
                "Please run 'python train_model.py' first."
            )

        # 1. Load the winning pipeline (used for primary classification & probability)
        self.best_pipeline = joblib.load(best_pipeline_path)

        # 2. Load the Logistic Regression pipeline (used for word-level contributions)
        self.logreg_pipeline = joblib.load(logreg_pipeline_path)

        # 3. Load metadata configuration
        if metadata_path.is_file():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata: Dict[str, Any] = json.load(f)
            self.model_name: str = self.metadata.get("best_model_name", "Logistic Regression")
            self.threshold: float = float(self.metadata.get("default_threshold", 0.5))
        else:
            self.model_name = "Logistic Regression"
            self.threshold = 0.5
            self.metadata = {}

        # 3b. Load data-driven risk cutoffs from metadata with fallback to defaults
        # Calibrated via calibrate_risk_levels.py on cross-validated training scores
        risk_cutoffs = self.metadata.get("risk_cutoffs", {})
        self.risk_low_max: float = float(risk_cutoffs.get("low_max", DEFAULT_RISK_LOW_MAX))
        self.risk_high_min: float = float(risk_cutoffs.get("high_min", DEFAULT_RISK_HIGH_MIN))

        # 4. Initialize rule-based red flag checker
        self.rule_checker = RuleChecker()

        # Cache Logistic Regression text vocabulary and coefficients for fast local explainability
        logreg_preprocessor = self.logreg_pipeline.named_steps["preprocessor"]
        logreg_classifier = self.logreg_pipeline.named_steps["classifier"]
        tfidf_trans = logreg_preprocessor.named_transformers_["text"]

        self._tfidf_vectorizer = tfidf_trans
        self._vocab_feature_names: List[str] = list(tfidf_trans.get_feature_names_out())
        self._logreg_coefs = logreg_classifier.coef_[0][:len(self._vocab_feature_names)]

    @staticmethod
    def label_for_level(risk_level: str) -> str:
        """
        Maps a calibrated risk level ('low', 'medium', 'high') to its descriptive
        classification output label.

        Why this matters for viva:
        Ensures consistent 3-tier classification across all views so borderline ads
        (e.g., 40%-49% risk score) are clearly identified as 'Borderline - review carefully'
        rather than mislabeled as 'Likely genuine'.

        Mapping Rules:
        - LOW    -> "Likely genuine"
        - MEDIUM -> "Borderline - review carefully"
        - HIGH   -> "Potentially fraudulent"

        Args:
            risk_level (str): 'low', 'medium', or 'high'.

        Returns:
            str: "Likely genuine", "Borderline - review carefully", or "Potentially fraudulent".
        """
        level = str(risk_level or "").strip().lower()
        if level == "low":
            return "Likely genuine"
        elif level == "medium":
            return "Borderline - review carefully"
        elif level == "high":
            return "Potentially fraudulent"
        return "Borderline - review carefully"

    # Alias for backwards compatibility across existing calls
    get_risk_label = label_for_level

    def predict(
        self,
        title: str = "",
        company_profile: str = "",
        description: str = "",
        requirements: str = "",
        benefits: str = "",
        salary_range: str = "",
        telecommuting: int = 0,
        has_company_logo: int = 1,
        has_questions: int = 1,
    ) -> Dict[str, Any]:
        """
        Performs full risk scoring on a candidate job advertisement.

        Order of Operations:
        1. Validates that job ad description text is provided.
        2. Engineers identical feature set to train_model.py (text cleaning, missingness flags).
        3. Generates probability using best_pipeline.
        4. Evaluates risk level (low/medium/high) and threshold classification label.
        5. Runs heuristic RuleChecker for red flags.
        6. Extracts top 5 words pushing toward fraud and top 5 words pushing toward genuine.

        Args:
            title (str): Job title.
            company_profile (str): Overview of company background.
            description (str): Primary job description body.
            requirements (str): Qualifications / skills needed.
            benefits (str): Perks, compensation, healthcare, 401k, etc.
            salary_range (str): Salary interval (e.g. "50000-80000").
            telecommuting (int): 1 if remote/telecommute, else 0.
            has_company_logo (int): 1 if employer logo present, else 0.
            has_questions (int): 1 if screening questions included, else 0.

        Returns:
            Dict[str, Any]: Comprehensive prediction response dictionary.

        Raises:
            ValueError: If description is empty or whitespace-only.
        """
        # ---------------------------------------------------------------------
        # Step 1: Input Validation
        # ---------------------------------------------------------------------
        if not description or not str(description).strip():
            raise ValueError("Please paste the job ad text")

        clean_title = str(title or "").strip()
        clean_profile = str(company_profile or "").strip()
        clean_desc = str(description or "").strip()
        clean_req = str(requirements or "").strip()
        clean_ben = str(benefits or "").strip()
        clean_salary = str(salary_range or "").strip()

        # ---------------------------------------------------------------------
        # Step 2: Feature Engineering (Identical to prepare_data.py)
        # ---------------------------------------------------------------------
        # Consolidated text column: title, company_profile, description, requirements, benefits
        combined_raw = f"{clean_title} {clean_profile} {clean_desc} {clean_req} {clean_ben}"
        cleaned_text = clean_text_content(combined_raw)

        # Flag 1: missing_company_profile (1 if empty/absent, else 0)
        missing_company_profile = 1 if not clean_profile or clean_profile.lower() == "nan" else 0

        # Flag 2: missing_salary (1 if empty/absent, else 0)
        missing_salary = 1 if not clean_salary or clean_salary.lower() == "nan" else 0

        # Flag 3: short_description (1 if description has fewer than 30 words, else 0)
        desc_word_count = len(clean_desc.split())
        short_description = 1 if desc_word_count < 30 else 0

        # Prepare single-row DataFrame matching the exact ColumnTransformer inputs
        input_row = pd.DataFrame([{
            "job_id": 0,
            "title": clean_title,
            "text": cleaned_text,
            "telecommuting": int(telecommuting),
            "has_company_logo": int(has_company_logo),
            "has_questions": int(has_questions),
            "missing_company_profile": int(missing_company_profile),
            "missing_salary": int(missing_salary),
            "short_description": int(short_description),
        }])

        # ---------------------------------------------------------------------
        # Step 3: Probability Estimation
        # ---------------------------------------------------------------------
        # Predict class probabilities: [P(Genuine), P(Fraudulent)]
        probs = self.best_pipeline.predict_proba(input_row)[0]
        fraud_probability: float = float(probs[1])

        # ---------------------------------------------------------------------
        # Step 4: Risk Level Assignment & Classification Label
        # ---------------------------------------------------------------------
        # Assign risk level based on empirically calibrated cutoffs from metadata.json
        if fraud_probability < self.risk_low_max:
            risk_level = "low"
        elif fraud_probability < self.risk_high_min:
            risk_level = "medium"
        else:
            risk_level = "high"

        # Classification label mapped directly from the calibrated risk level:
        # LOW    -> "Likely genuine"
        # MEDIUM -> "Borderline - review carefully"
        # HIGH   -> "Potentially fraudulent"
        label: str = self.label_for_level(risk_level)

        # ---------------------------------------------------------------------
        # Step 5: Heuristic Rule Warnings
        # ---------------------------------------------------------------------
        warnings = self.rule_checker.check(
            title=clean_title,
            company_profile=clean_profile,
            text=clean_desc,
            salary_range=clean_salary,
        )

        # ---------------------------------------------------------------------
        # Step 6: Word Contributions (Local Explainability via Logistic Regression)
        # ---------------------------------------------------------------------
        top_words = self._explain_text(cleaned_text)

        # ---------------------------------------------------------------------
        # Step 7: Return Standardized Advisory Response
        # Note: Probability is an estimate, never proof.
        # ---------------------------------------------------------------------
        return {
            "fraud_probability": round(fraud_probability, 4),
            "label": label,
            "risk_level": risk_level,
            "warnings": warnings,
            "top_words": top_words,
            "model_name": self.model_name,
            "threshold": self.threshold,
            "risk_cutoffs": {
                "low_max": self.risk_low_max,
                "high_min": self.risk_high_min,
            },
            "features_used": {
                "word_count": len(cleaned_text.split()),
                "missing_company_profile": bool(missing_company_profile),
                "missing_salary": bool(missing_salary),
                "short_description": bool(short_description),
                "telecommuting": bool(telecommuting),
                "has_company_logo": bool(has_company_logo),
                "has_questions": bool(has_questions),
            },
        }

    def _explain_text(self, cleaned_text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculates local word contributions for words appearing in this job ad.

        Formula:
            Contribution = TF-IDF(word) * Logistic_Regression_Coefficient(word)

        Args:
            cleaned_text (str): Cleaned and normalized text of the ad.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Top 5 fraud-pushing and top 5 genuine-pushing words.
        """
        tfidf_row = self._tfidf_vectorizer.transform([cleaned_text])
        nz_indices = tfidf_row.indices
        nz_values = tfidf_row.data

        word_contributions: List[Dict[str, Any]] = []

        for idx, tfidf_val in zip(nz_indices, nz_values):
            word = self._vocab_feature_names[idx]
            coef = float(self._logreg_coefs[idx])
            contribution = float(tfidf_val * coef)

            word_contributions.append({
                "word": word,
                "contribution": round(contribution, 4),
                "tfidf": round(float(tfidf_val), 4),
                "coef": round(coef, 4),
            })

        if not word_contributions:
            return {"fraud": [], "genuine": []}

        # Pushing toward fraud: highest positive contribution values
        fraud_pushing = sorted(
            [item for item in word_contributions if item["contribution"] > 0],
            key=lambda x: x["contribution"],
            reverse=True,
        )[:5]

        # Pushing toward genuine: lowest negative contribution values (most negative)
        genuine_pushing = sorted(
            [item for item in word_contributions if item["contribution"] < 0],
            key=lambda x: x["contribution"],
        )[:5]

        return {
            "fraud": fraud_pushing,
            "genuine": genuine_pushing,
        }


# Canonical function and backward-compatible alias for module-level import
label_for_level = Predictor.label_for_level
get_risk_label = Predictor.label_for_level


def format_demo_output(demo_num: int, title_desc: str, result: Dict[str, Any]) -> None:
    """Helper to display demo prediction results cleanly in terminal output."""
    print("=" * 80)
    print(f"DEMO AD #{demo_num}: {title_desc}")
    print("=" * 80)
    print(f"Model Used:          {result['model_name']} (Threshold: {result['threshold']})")
    print(f"Fraud Probability:   {result['fraud_probability'] * 100:.2f}%")
    print(f"Risk Level:          {result['risk_level'].upper()}")
    print(f"Predicted Label:     {result['label']}")
    print(f"Word Count:          {result['features_used']['word_count']} words")

    print("\nHeuristic Red Flag Warnings:")
    if result["warnings"]:
        for w in result["warnings"]:
            print(f"  - [{w['severity'].upper()}] {w['message']} (Matched: {w['matched']})")
    else:
        print("  - None triggered. All heuristic checks passed cleanly.")

    print("\nTop Words Influencing Prediction (Logistic Regression):")
    print("  Pushing Toward FRAUD:")
    if result["top_words"]["fraud"]:
        for item in result["top_words"]["fraud"]:
            print(f"    * '{item['word']}': contribution = +{item['contribution']:.4f} "
                  f"(TF-IDF: {item['tfidf']:.3f}, Coef: {item['coef']:.2f})")
    else:
        print("    * (No strong fraud-pushing vocabulary present)")

    print("  Pushing Toward GENUINE:")
    if result["top_words"]["genuine"]:
        for item in result["top_words"]["genuine"]:
            print(f"    * '{item['word']}': contribution = {item['contribution']:.4f} "
                  f"(TF-IDF: {item['tfidf']:.3f}, Coef: {item['coef']:.2f})")
    else:
        print("    * (No strong genuine-pushing vocabulary present)")

    print("=" * 80 + "\n")


def run_demo_ads() -> None:
    """
    Executes and displays predictions for three representative demo job postings:
    1. Realistic genuine software internship ad.
    2. Blatant scam ad with fee requests, WhatsApp contact, and high earnings.
    3. Vague short ad with no company profile and minimal information.
    """
    print("\n" + "#" * 80)
    print("RUNNING PREDICTOR DEMO WITH 3 SAMPLE ADVERTISEMENTS")
    print("#" * 80 + "\n")

    predictor = Predictor()

    # -------------------------------------------------------------------------
    # Demo Ad 1: Realistic Genuine Software Internship Ad
    # -------------------------------------------------------------------------
    ad_1_result = predictor.predict(
        title="Software Engineering Intern (Summer 2026)",
        company_profile=(
            "Apex Cloud Solutions is an enterprise software consultancy specializing in "
            "distributed microservices, cloud migrations, and real-time data pipelines. "
            "Founded in 2014, we pride ourselves on collaborative mentorship, code quality, "
            "and active open-source contribution."
        ),
        description=(
            "We are looking for an ambitious Software Engineering Intern to join our engineering "
            "team in New York. You will collaborate directly with senior engineers to design RESTful "
            "APIs in Python and FastAPI, write automated unit tests, and deploy services to Kubernetes. "
            "This is a hands-on learning internship with dedicated 1-on-1 mentorship."
        ),
        requirements=(
            "Pursuing a BS or MS in Computer Science, Software Engineering, or related technical field. "
            "Solid understanding of object-oriented programming in Python, Git version control, "
            "and relational database concepts. Strong written and verbal communication skills."
        ),
        benefits=(
            "Competitive hourly salary ($38-$45/hr), mentorship program, full health benefits, "
            "flexible hybrid work schedule, weekly team lunches, and potential full-time return offer."
        ),
        salary_range="75000-90000",
        telecommuting=0,
        has_company_logo=1,
        has_questions=1,
    )
    format_demo_output(1, "Realistic Genuine Software Internship Ad", ad_1_result)

    # -------------------------------------------------------------------------
    # Demo Ad 2: Scam-Style Ad
    # -------------------------------------------------------------------------
    ad_2_result = predictor.predict(
        title="Urgent Data Entry Assistant - Immediate Joining!",
        company_profile="",  # Missing company profile
        description=(
            "Urgent hiring! Work from home and earn $1500 per week guaranteed income! "
            "No experience required. Immediate joining available for limited positions. "
            "Send money for refundable registration fee and background check deposit. "
            "Contact recruiter directly on WhatsApp or personal email at jobrecruiter2026@gmail.com."
        ),
        requirements="No experience required. Laptop and internet connection only.",
        benefits="Get rich quick with daily cash payouts.",
        salary_range="",     # Missing salary
        telecommuting=1,
        has_company_logo=0,
        has_questions=0,
    )
    format_demo_output(2, "Blatant Scam Ad (Registration Fee & WhatsApp)", ad_2_result)

    # -------------------------------------------------------------------------
    # Demo Ad 3: Vague Short Ad
    # -------------------------------------------------------------------------
    ad_3_result = predictor.predict(
        title="Administrative Assistant",
        company_profile="",  # Missing company profile
        description="Looking for an office assistant to do basic clerical tasks immediately. Good pay for right candidate.",
        requirements="",
        benefits="",
        salary_range="",     # Missing salary
        telecommuting=0,
        has_company_logo=0,
        has_questions=0,
    )
    format_demo_output(3, "Vague Short Ad (No Profile & Minimal Details)", ad_3_result)


if __name__ == "__main__":
    run_demo_ads()
