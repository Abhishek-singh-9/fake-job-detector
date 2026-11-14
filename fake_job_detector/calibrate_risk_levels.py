"""
Risk Level Cutoff Calibration Module for Fake Job Posting Detector.

This script determines data-driven risk level thresholds (LOW, MEDIUM, HIGH)
for the job posting fraud detection pipeline.

Key Machine Learning Principles & Viva Points:
- Why Out-of-Fold (OOF) Scores on the Train Set?
  1. No Test-Set Peeking / Zero Data Leakage:
     The test set must be reserved strictly for final, unbiased performance estimation.
     If candidate cutoffs were selected by optimizing metrics on the test set, the thresholds
     would overfit to that specific test sample, creating data leakage and overly optimistic
     confidence.
  2. Realistic Probabilities via 5-Fold Stratified Cross-Validation:
     Evaluating a model on the exact data it was trained on produces overfitted, overconfident
     probabilities (scores compressed excessively toward 0.0 and 1.0).
     Using `cross_val_predict(..., method='predict_proba')` generates genuine out-of-fold
     predictions: every ad in the training set is scored ONLY by an estimator fold that did
     not see that ad during training. This mirrors live inference on unseen candidates.
- Optimization Criteria:
  - LOW_MAX: Ensures high sensitivity/recall for fraud (>= 90% recall), so ads classified as
    LOW risk rarely conceal a real scam.
  - HIGH_MIN: Ensures acceptable precision (>= 50%), so ads classified as HIGH risk have high
    veracity and do not overwhelm users with false alarms.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Fixed random seed for complete reproducibility
RANDOM_STATE: int = 42

# Candidate probability cutoffs to evaluate
CANDIDATES: List[float] = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# Target calibration rules
MIN_RECALL_FOR_LOW_MAX: float = 0.90      # At least 90% of scams must score >= LOW_MAX
MIN_PRECISION_FOR_HIGH_MIN: float = 0.50  # At least 50% precision for HIGH_MIN


def get_project_paths() -> Tuple[Path, Path, Path, Path]:
    """
    Resolves project directories and key file paths relative to this script.

    Returns:
        Tuple[Path, Path, Path, Path]: (train_path, test_path, pipeline_path, metadata_path)
    """
    script_dir: Path = Path(__file__).resolve().parent
    train_path: Path = script_dir / "data" / "processed" / "train.csv"
    test_path: Path = script_dir / "data" / "processed" / "test.csv"
    pipeline_path: Path = script_dir / "models" / "best_pipeline.joblib"
    metadata_path: Path = script_dir / "models" / "metadata.json"

    return train_path, test_path, pipeline_path, metadata_path


def load_data_and_model() -> Tuple[pd.DataFrame, pd.DataFrame, Any, Dict[str, Any]]:
    """
    Loads train and test datasets, the trained best_pipeline model, and metadata.

    Why this matters:
    Ensures missing values in the 'text' column are filled with empty strings so the
    TF-IDF text transformer receives valid string data identically to training.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Any, Dict[str, Any]]:
            (train_df, test_df, best_pipeline, metadata_dict)
    """
    train_path, test_path, pipeline_path, metadata_path = get_project_paths()

    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Processed data files not found at {train_path} or {test_path}. "
            "Please run 'prepare_data.py' first."
        )

    if not pipeline_path.is_file():
        raise FileNotFoundError(
            f"Trained model pipeline not found at {pipeline_path}. "
            "Please run 'train_model.py' first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Impute missing text values with empty strings
    train_df["text"] = train_df["text"].fillna("")
    test_df["text"] = test_df["text"].fillna("")

    best_pipeline = joblib.load(pipeline_path)

    metadata: Dict[str, Any] = {}
    if metadata_path.is_file():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return train_df, test_df, best_pipeline, metadata


def compute_out_of_fold_scores(
    train_df: pd.DataFrame,
    pipeline: Any,
) -> np.ndarray:
    """
    Computes out-of-fold predicted fraud probabilities across 5 Stratified folds on TRAIN.

    Why out-of-fold scores are used (Essential Viva Principle):
    - To avoid tuning cutoffs on the test set (which would cause data leakage).
    - To avoid self-evaluation bias on the training set (in-sample predictions are overfitted).
    - By cloning the pipeline and evaluating each sample strictly with estimators trained
      without that sample, we get realistic out-of-sample probability estimates.

    Args:
        train_df (pd.DataFrame): Training set DataFrame.
        pipeline (Any): Fitted or unfitted Pipeline instance to clone.

    Returns:
        np.ndarray: 1D array of out-of-fold fraud probabilities for each row in train_df.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fresh_pipeline = clone(pipeline)

    print("Generating 5-Fold Stratified out-of-fold probability scores on TRAIN...")
    # method='predict_proba' returns [P(Genuine), P(Fraudulent)]
    oof_proba = cross_val_predict(
        estimator=fresh_pipeline,
        X=train_df,
        y=train_df["fraudulent"],
        cv=cv,
        method="predict_proba",
        n_jobs=1,  # Safe single-process execution for Windows environments
    )

    # Column index 1 represents probability of positive class (fraudulent)
    return oof_proba[:, 1]


def build_candidate_metrics_table(
    y_true: np.ndarray,
    scores: np.ndarray,
    candidates: List[float],
) -> pd.DataFrame:
    """
    Computes genuine ad coverage below cutoff, fraud recall, and precision for each cutoff.

    Metrics calculated per cutoff:
    1. % Genuine < Cutoff: Genuine ads with score < cutoff divided by total genuine ads.
    2. % Fraud >= Cutoff (Recall): Fraud ads with score >= cutoff divided by total fraud ads.
    3. Precision (%): Fraud ads with score >= cutoff divided by all ads with score >= cutoff.

    Args:
        y_true (np.ndarray): Binary labels (0 = Genuine, 1 = Fraudulent).
        scores (np.ndarray): Predicted fraud probability scores.
        candidates (List[float]): Candidate probability cutoffs.

    Returns:
        pd.DataFrame: Table with evaluation metrics for each candidate cutoff.
    """
    n_genuine: int = int((y_true == 0).sum())
    n_fraud: int = int((y_true == 1).sum())

    records: List[Dict[str, Any]] = []

    for cutoff in candidates:
        # Genuine ads scoring strictly below the cutoff
        genuine_below_count = int(((y_true == 0) & (scores < cutoff)).sum())
        genuine_below_pct = (genuine_below_count / n_genuine) * 100.0 if n_genuine > 0 else 0.0

        # Fraudulent ads scoring at or above the cutoff (Recall)
        fraud_above_count = int(((y_true == 1) & (scores >= cutoff)).sum())
        fraud_recall_pct = (fraud_above_count / n_fraud) * 100.0 if n_fraud > 0 else 0.0

        # Total ads scoring at or above cutoff
        total_above_count = int((scores >= cutoff).sum())
        precision_pct = (
            (fraud_above_count / total_above_count) * 100.0
            if total_above_count > 0
            else 0.0
        )

        records.append({
            "cutoff": cutoff,
            "genuine_below_count": genuine_below_count,
            "genuine_below_pct": genuine_below_pct,
            "fraud_above_count": fraud_above_count,
            "fraud_recall_pct": fraud_recall_pct,
            "total_above_count": total_above_count,
            "precision_pct": precision_pct,
        })

    return pd.DataFrame(records)


def select_calibrated_cutoffs(
    metrics_df: pd.DataFrame,
    candidates: List[float],
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Chooses optimal LOW_MAX and HIGH_MIN cutoffs according to project rules:

    Rule 1 (LOW_MAX):
      The HIGHEST candidate cutoff at which at least 90% of fraudulent ads still
      score at or above it (so a LOW result rarely hides a fake ad).
    Rule 2 (HIGH_MIN):
      The LOWEST candidate cutoff at which precision is at least 50%.
      If none reaches 50%, use the candidate cutoff with the highest precision.
    Rule 3 (Order Consistency Check):
      If LOW_MAX >= HIGH_MIN, issue a warning and set LOW_MAX to the next lower candidate.

    Args:
        metrics_df (pd.DataFrame): Table of candidate cutoff metrics.
        candidates (List[float]): Sorted candidate cutoffs.

    Returns:
        Tuple[float, float, Dict[str, Any]]: (chosen_low_max, chosen_high_min, explanation_dict)
    """
    # -------------------------------------------------------------------------
    # Rule 1: Choose LOW_MAX
    # -------------------------------------------------------------------------
    # Candidates where fraud recall is >= 90% (MIN_RECALL_FOR_LOW_MAX * 100)
    target_recall_pct = MIN_RECALL_FOR_LOW_MAX * 100.0
    qualifying_low = metrics_df[metrics_df["fraud_recall_pct"] >= target_recall_pct]

    if not qualifying_low.empty:
        # Highest candidate cutoff among those meeting the recall threshold
        low_max_row = qualifying_low.sort_values(by="cutoff", ascending=False).iloc[0]
        low_max: float = float(low_max_row["cutoff"])
        low_max_reason: str = (
            f"Highest cutoff with Fraud Recall >= {target_recall_pct:.1f}% "
            f"(Achieved: {low_max_row['fraud_recall_pct']:.2f}% recall, "
            f"{low_max_row['genuine_below_pct']:.2f}% genuine ads below cutoff)."
        )
    else:
        # Fallback to minimum candidate if none achieved 90%
        low_max = float(candidates[0])
        low_max_reason = (
            f"No candidate reached {target_recall_pct:.1f}% recall. "
            f"Fell back to minimum candidate {low_max:.2f}."
        )

    # -------------------------------------------------------------------------
    # Rule 2: Choose HIGH_MIN
    # -------------------------------------------------------------------------
    # Candidates where precision is >= 50% (MIN_PRECISION_FOR_HIGH_MIN * 100)
    target_prec_pct = MIN_PRECISION_FOR_HIGH_MIN * 100.0
    qualifying_high = metrics_df[metrics_df["precision_pct"] >= target_prec_pct]

    if not qualifying_high.empty:
        # Lowest candidate cutoff achieving at least 50% precision
        high_min_row = qualifying_high.sort_values(by="cutoff", ascending=True).iloc[0]
        high_min: float = float(high_min_row["cutoff"])
        high_min_reason: str = (
            f"Lowest cutoff with Precision >= {target_prec_pct:.1f}% "
            f"(Achieved: {high_min_row['precision_pct']:.2f}% precision, "
            f"{high_min_row['fraud_recall_pct']:.2f}% recall)."
        )
    else:
        # If none reaches 50%, pick candidate with highest precision
        best_prec_row = metrics_df.sort_values(by="precision_pct", ascending=False).iloc[0]
        high_min = float(best_prec_row["cutoff"])
        high_min_reason = (
            f"Notice: No candidate reached {target_prec_pct:.1f}% precision! "
            f"Selected cutoff with best overall precision: {high_min:.2f} "
            f"({best_prec_row['precision_pct']:.2f}%)."
        )

    # -------------------------------------------------------------------------
    # Rule 3: Consistency Check (LOW_MAX < HIGH_MIN)
    # -------------------------------------------------------------------------
    adjusted_warning: str = ""
    if low_max >= high_min:
        current_idx = candidates.index(low_max)
        if current_idx > 0:
            previous_cutoff = candidates[current_idx - 1]
            adjusted_warning = (
                f"WARNING: LOW_MAX ({low_max:.2f}) >= HIGH_MIN ({high_min:.2f}). "
                f"Per rule, adjusting LOW_MAX down to next lower candidate: {previous_cutoff:.2f}."
            )
            low_max = previous_cutoff
        else:
            adjusted_warning = (
                f"WARNING: LOW_MAX ({low_max:.2f}) >= HIGH_MIN ({high_min:.2f}), "
                "but LOW_MAX is already at lowest candidate."
            )

    explanations: Dict[str, Any] = {
        "low_max": low_max,
        "high_min": high_min,
        "low_max_reason": low_max_reason,
        "high_min_reason": high_min_reason,
        "adjusted_warning": adjusted_warning,
    }

    return low_max, high_min, explanations


def evaluate_test_set_buckets(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    low_max: float,
    high_min: float,
) -> pd.DataFrame:
    """
    Evaluates how test set genuine and fraudulent ads distribute across the calibrated tiers:
    - LOW:    score < low_max
    - MEDIUM: low_max <= score < high_min
    - HIGH:   score >= high_min

    Args:
        y_test (np.ndarray): Ground truth binary test labels.
        test_scores (np.ndarray): Model-predicted fraud probabilities on test set.
        low_max (float): Calibrated low-risk upper boundary.
        high_min (float): Calibrated high-risk lower boundary.

    Returns:
        pd.DataFrame: Breakdown dataframe with counts and percentages per bucket.
    """
    total_genuine: int = int((y_test == 0).sum())
    total_fraud: int = int((y_test == 1).sum())
    total_ads: int = len(y_test)

    # Masks for each risk interval
    low_mask = test_scores < low_max
    med_mask = (test_scores >= low_max) & (test_scores < high_min)
    high_mask = test_scores >= high_min

    tiers = [
        ("LOW", f"Score < {low_max:.2f}", low_mask),
        ("MEDIUM", f"{low_max:.2f} <= Score < {high_min:.2f}", med_mask),
        ("HIGH", f"Score >= {high_min:.2f}", high_mask),
    ]

    records: List[Dict[str, Any]] = []

    for name, score_range, mask in tiers:
        gen_cnt = int(((y_test == 0) & mask).sum())
        gen_pct = (gen_cnt / total_genuine) * 100.0 if total_genuine > 0 else 0.0

        fraud_cnt = int(((y_test == 1) & mask).sum())
        fraud_pct = (fraud_cnt / total_fraud) * 100.0 if total_fraud > 0 else 0.0

        bucket_total = int(mask.sum())
        fraud_share_pct = (fraud_cnt / bucket_total) * 100.0 if bucket_total > 0 else 0.0

        records.append({
            "Risk Tier": name,
            "Score Interval": score_range,
            "Genuine Ads": f"{gen_cnt:,} ({gen_pct:.2f}%)",
            "Fraudulent Ads": f"{fraud_cnt:,} ({fraud_pct:.2f}%)",
            "Total Ads": f"{bucket_total:,} ({bucket_total / total_ads * 100:.2f}%)",
            "Fraud Share in Tier": f"{fraud_share_pct:.2f}% ({fraud_cnt}/{bucket_total:,})",
            "raw_gen_cnt": gen_cnt,
            "raw_fraud_cnt": fraud_cnt,
            "raw_total": bucket_total,
        })

    return pd.DataFrame(records)


def save_cutoffs_to_metadata(
    metadata_path: Path,
    metadata: Dict[str, Any],
    low_max: float,
    high_min: float,
) -> None:
    """
    Appends the chosen risk cutoffs into models/metadata.json without deleting any existing key.

    Args:
        metadata_path (Path): Path to metadata.json.
        metadata (Dict[str, Any]): Existing metadata dictionary.
        low_max (float): Calibrated low cutoff.
        high_min (float): Calibrated high cutoff.
    """
    metadata["risk_cutoffs"] = {
        "low_max": round(float(low_max), 2),
        "high_min": round(float(high_min), 2),
    }
    metadata["risk_cutoffs_calibrated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nSaved updated cutoffs to: {metadata_path}")


def print_formatted_tables(
    train_metrics_df: pd.DataFrame,
    explanations: Dict[str, Any],
    test_metrics_df: pd.DataFrame,
    test_buckets_df: pd.DataFrame,
) -> None:
    """Helper function to print all diagnostic tables and decision rationale cleanly."""
    print("\n" + "=" * 92)
    print("STEP 1 & 2: OUT-OF-FOLD EVALUATION ON TRAIN SET (5-FOLD STRATIFIED CV)")
    print("=" * 92)
    print("Why OOF on TRAIN? To calibrate cutoffs using realistic probabilities without")
    print("contaminating or tuning on the test set.")
    print("-" * 92)
    print(
        f"{'Cutoff':<8} "
        f"{'% Genuine < Cutoff':<22} "
        f"{'% Fraud >= Cutoff (Recall)':<28} "
        f"{'Precision (%) [Fraud Share]':<28}"
    )
    print("-" * 92)

    for _, row in train_metrics_df.iterrows():
        print(
            f"{row['cutoff']:<8.2f} "
            f"{row['genuine_below_pct']:>6.2f}% ({int(row['genuine_below_count']):>5} ads)        "
            f"{row['fraud_recall_pct']:>6.2f}% ({int(row['fraud_above_count']):>3} scams)          "
            f"{row['precision_pct']:>6.2f}% ({int(row['fraud_above_count'])}/{int(row['total_above_count'])})"
        )
    print("=" * 92)

    # -------------------------------------------------------------------------
    # Decision Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("STEP 4: CUTOFF SELECTION BY DEFINED DECISION RULES")
    print("=" * 92)
    print(f"Rule 1 (LOW_MAX):  Pick highest cutoff where Fraud Recall >= {MIN_RECALL_FOR_LOW_MAX*100:.0f}%.")
    print(f"  -> Chosen LOW_MAX  = {explanations['low_max']:.2f}")
    print(f"  -> Explanation:     {explanations['low_max_reason']}")
    print()
    print(f"Rule 2 (HIGH_MIN): Pick lowest cutoff where Precision >= {MIN_PRECISION_FOR_HIGH_MIN*100:.0f}%.")
    print(f"  -> Chosen HIGH_MIN = {explanations['high_min']:.2f}")
    print(f"  -> Explanation:     {explanations['high_min_reason']}")

    if explanations["adjusted_warning"]:
        print()
        print(f"Rule 3 (Order Consistency Check):")
        print(f"  -> {explanations['adjusted_warning']}")
    else:
        print()
        print(f"Rule 3 (Order Consistency Check):")
        print(f"  -> Valid order confirmed: LOW_MAX ({explanations['low_max']:.2f}) < HIGH_MIN ({explanations['high_min']:.2f}).")

    print("-" * 92)
    print(f"FINAL CALIBRATED TIERS:")
    print(f"  - LOW RISK:    Probability strictly below {explanations['low_max']:.2f} (< {explanations['low_max']*100:.0f}%)")
    print(f"  - MEDIUM RISK: Probability from {explanations['low_max']:.2f} to < {explanations['high_min']:.2f} ({explanations['low_max']*100:.0f}% to < {explanations['high_min']*100:.0f}%)")
    print(f"  - HIGH RISK:   Probability at or above {explanations['high_min']:.2f} (>= {explanations['high_min']*100:.0f}%)")
    print("=" * 92)

    # -------------------------------------------------------------------------
    # Test Set Candidate Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("STEP 5A: CANDIDATE CUTOFFS EVALUATED ON UNSEEN TEST SET")
    print("=" * 92)
    print(
        f"{'Cutoff':<8} "
        f"{'% Genuine < Cutoff':<22} "
        f"{'% Fraud >= Cutoff (Recall)':<28} "
        f"{'Precision (%) [Fraud Share]':<28}"
    )
    print("-" * 92)
    for _, row in test_metrics_df.iterrows():
        print(
            f"{row['cutoff']:<8.2f} "
            f"{row['genuine_below_pct']:>6.2f}% ({int(row['genuine_below_count']):>5} ads)        "
            f"{row['fraud_recall_pct']:>6.2f}% ({int(row['fraud_above_count']):>3} scams)          "
            f"{row['precision_pct']:>6.2f}% ({int(row['fraud_above_count'])}/{int(row['total_above_count'])})"
        )
    print("=" * 92)

    # -------------------------------------------------------------------------
    # Test Set Risk Tier Breakdown
    # -------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"STEP 5B: UNSEEN TEST SET DISTRIBUTION ACROSS CALIBRATED RISK TIERS")
    print(f"(LOW_MAX = {explanations['low_max']:.2f}, HIGH_MIN = {explanations['high_min']:.2f})")
    print("=" * 92)
    display_cols = ["Risk Tier", "Score Interval", "Genuine Ads", "Fraudulent Ads", "Total Ads", "Fraud Share in Tier"]
    print(test_buckets_df[display_cols].to_string(index=False))
    print("=" * 92)


def main() -> None:
    """
    Main orchestration routine for risk level threshold calibration.
    """
    print("\n" + "#" * 92)
    print("CALIBRATING DATA-DRIVEN RISK LEVEL CUTOFFS")
    print("#" * 92)

    # 1. Load data, models, and metadata
    train_path, test_path, pipeline_path, metadata_path = get_project_paths()
    train_df, test_df, best_pipeline, metadata = load_data_and_model()

    y_train = train_df["fraudulent"].values
    y_test = test_df["fraudulent"].values

    print(f"Loaded training data: {len(train_df):,} samples (Fraud: {(y_train == 1).sum()}, Genuine: {(y_train == 0).sum()})")
    print(f"Loaded test data:     {len(test_df):,} samples (Fraud: {(y_test == 1).sum()}, Genuine: {(y_test == 0).sum()})")
    print(f"Loaded model:         {metadata.get('best_model_name', 'Trained Pipeline')} from {pipeline_path.name}")

    # 2. Get out-of-fold fraud scores on TRAIN set with 5-fold StratifiedKFold
    oof_scores = compute_out_of_fold_scores(train_df=train_df, pipeline=best_pipeline)

    # 3. Print table for candidate cutoffs using out-of-fold scores
    train_metrics_df = build_candidate_metrics_table(
        y_true=y_train,
        scores=oof_scores,
        candidates=CANDIDATES,
    )

    # 4. Choose cutoffs by specified decision rules
    low_max, high_min, explanations = select_calibrated_cutoffs(
        metrics_df=train_metrics_df,
        candidates=CANDIDATES,
    )

    # 5. Evaluate on unseen TEST set using chosen cutoffs
    test_scores = best_pipeline.predict_proba(test_df)[:, 1]

    test_metrics_df = build_candidate_metrics_table(
        y_true=y_test,
        scores=test_scores,
        candidates=CANDIDATES,
    )

    test_buckets_df = evaluate_test_set_buckets(
        y_test=y_test,
        test_scores=test_scores,
        low_max=low_max,
        high_min=high_min,
    )

    # Print clean terminal reports
    print_formatted_tables(
        train_metrics_df=train_metrics_df,
        explanations=explanations,
        test_metrics_df=test_metrics_df,
        test_buckets_df=test_buckets_df,
    )

    # 6. Save chosen values into models/metadata.json without removing existing keys
    save_cutoffs_to_metadata(
        metadata_path=metadata_path,
        metadata=metadata,
        low_max=low_max,
        high_min=high_min,
    )

    print("\nCalibration successfully completed and saved.\n")


if __name__ == "__main__":
    main()
