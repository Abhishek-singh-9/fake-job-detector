"""
Model Training & Evaluation Pipeline for Fake Job Posting Detector.

This script trains, evaluates, and compares three machine learning classification models:
1. Logistic Regression (balanced class weights, linear baseline, explainable coefficients)
2. Multinomial Naive Bayes (text-only probabilistic baseline)
3. Random Forest (ensemble of decision trees with balanced subsampling)

Key Machine Learning Principles (Essential for Viva):
- Data Leakage Prevention:
  All TF-IDF vectorization and feature processing is wrapped inside scikit-learn
  Pipelines. Vocabulary and inverse document frequencies are fitted STRICTLY on
  the training folds, never touching the validation folds or test set.
- Why Class Weighting instead of SMOTE for TF-IDF Text:
  1. High Dimensionality & Sparsity: TF-IDF has 10,000 sparse dimensions. SMOTE generates
     synthetic points by interpolating between nearest neighbors in euclidean space.
     In high-dimensional sparse text, euclidean distance degrades ("curse of dimensionality"),
     creating synthetic feature vectors that represent impossible word combinations.
  2. Loss Penalty vs. Data Corruption: Class weighting ('balanced' or 'balanced_subsample')
     directly alters the optimization objective by penalizing minority misclassifications
     proportionately to class frequencies, achieving robust cost-sensitive learning without
     corrupting real language features.
- Metric Selection:
  With ~4.5% fraud prevalence, accuracy is fundamentally misleading (a naive model predicting
  all 0s achieves ~95.5% accuracy). We prioritize F1-Score, Recall, Precision, and
  Precision-Recall AUC (PR-AUC).
- Train/Test Protocol:
  Model selection is decided SOLELY using 5-Fold Stratified Cross-Validation on the
  train set. The test set is evaluated once at the very end.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Fixed random seed for complete reproducibility across models and cross-validation
RANDOM_STATE: int = 42

# Structured flag columns engineered during data preparation
FLAG_COLUMNS: List[str] = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "missing_company_profile",
    "missing_salary",
    "short_description",
]

# Classification thresholds to evaluate for decision-making trade-offs
THRESHOLDS: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]


def get_project_paths() -> Tuple[Path, Path, Path, Path]:
    """
    Resolves directories and file paths relative to this script.

    Returns:
        Tuple[Path, Path, Path, Path]: (train_path, test_path, models_dir, reports_dir)
    """
    script_dir: Path = Path(__file__).resolve().parent
    train_path: Path = script_dir / "data" / "processed" / "train.csv"
    test_path: Path = script_dir / "data" / "processed" / "test.csv"
    models_dir: Path = script_dir / "models"
    reports_dir: Path = script_dir / "reports"

    # Ensure output directories exist
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return train_path, test_path, models_dir, reports_dir


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train and test datasets from data/processed/.

    Why this matters:
    Ensures missing values in 'text' are safely converted to empty strings so
    the TF-IDF vectorizer receives valid string inputs.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    train_path, test_path, _, _ = get_project_paths()

    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Processed data not found at {train_path} or {test_path}. "
            "Please run 'python prepare_data.py' first."
        )

    print(f"Loading training data from: {train_path.name}")
    train_df: pd.DataFrame = pd.read_csv(train_path)

    print(f"Loading test data from:     {test_path.name}")
    test_df: pd.DataFrame = pd.read_csv(test_path)

    # Impute missing text as empty strings to prevent vectorization errors
    train_df["text"] = train_df["text"].fillna("")
    test_df["text"] = test_df["text"].fillna("")

    return train_df, test_df


def build_model_pipelines() -> Dict[str, Pipeline]:
    """
    Constructs the model pipelines with appropriate ColumnTransformers.

    Why this matters for viva:
    - Encapsulating preprocessors and classifiers inside scikit-learn Pipelines guarantees
      that TF-IDF vocabulary and IDF calculations are strictly fit on training splits,
      completely eliminating test-set leakage.
    - Naive Bayes uses ONLY text features because MultinomialNB is formulated for word
      count/frequency statistics, while arbitrary binary flags violate its distribution.
    - Logistic Regression and Random Forest ingest both the TF-IDF text matrix and the
      6 binary flag features.

    Returns:
        Dict[str, Pipeline]: Dictionary mapping model names to configured Pipelines.
    """
    # Preprocessor for models utilizing both text and binary flags
    preprocessor_all = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=10000,
                    min_df=3,
                    stop_words="english",
                    sublinear_tf=True,
                ),
                "text",
            ),
            ("flags", "passthrough", FLAG_COLUMNS),
        ],
        remainder="drop",
    )

    # Preprocessor for Naive Bayes utilizing text features only
    preprocessor_text_only = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=10000,
                    min_df=3,
                    stop_words="english",
                    sublinear_tf=True,
                ),
                "text",
            ),
        ],
        remainder="drop",
    )

    pipelines: Dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor_all),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Multinomial Naive Bayes": Pipeline(
            steps=[
                ("preprocessor", preprocessor_text_only),
                ("classifier", MultinomialNB()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor_all),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return pipelines


def run_cross_validation(
    name: str,
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    cv_splitter: StratifiedKFold,
) -> Dict[str, float]:
    """
    Executes 5-fold Stratified Cross-Validation on the training set for a model.

    Why this matters for viva:
    Model selection must be based on out-of-fold generalization error on the training
    data, never by peeking at the test set.

    Args:
        name (str): Model name.
        pipeline (Pipeline): Model pipeline.
        train_df (pd.DataFrame): Training DataFrame.
        cv_splitter (StratifiedKFold): Stratified K-fold splitter.

    Returns:
        Dict[str, float]: Mean and std for precision, recall, f1, and pr_auc.
    """
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    pr_aucs: List[float] = []

    y_train = train_df["fraudulent"].values

    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(train_df, y_train), start=1):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]
        y_val = y_train[val_idx]

        # Clone and fit pipeline on this training fold only
        pipeline.fit(fold_train, fold_train["fraudulent"])

        # Predict labels and class probabilities on validation fold
        y_pred = pipeline.predict(fold_val)
        y_prob = pipeline.predict_proba(fold_val)[:, 1]

        # Calculate metrics for the positive (fraud) class
        prec = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        pr_auc = average_precision_score(y_val, y_prob)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        pr_aucs.append(pr_auc)

    return {
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_f1_std": float(np.std(f1s)),
        "cv_precision_mean": float(np.mean(precisions)),
        "cv_precision_std": float(np.std(precisions)),
        "cv_recall_mean": float(np.mean(recalls)),
        "cv_recall_std": float(np.std(recalls)),
        "cv_pr_auc_mean": float(np.mean(pr_aucs)),
        "cv_pr_auc_std": float(np.std(pr_aucs)),
    }


def evaluate_on_test_set(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Fits the pipeline on the ENTIRE training set and evaluates on the unseen test set.

    Args:
        pipeline (Pipeline): Model pipeline to fit and evaluate.
        train_df (pd.DataFrame): Complete training DataFrame.
        test_df (pd.DataFrame): Unseen test DataFrame.

    Returns:
        Dict[str, Any]: Dictionary containing test metrics, predictions, probabilities, and confusion matrix.
    """
    # Fit pipeline on the full training dataset
    pipeline.fit(train_df, train_df["fraudulent"])

    y_test = test_df["fraudulent"].values
    y_pred = pipeline.predict(test_df)
    y_prob = pipeline.predict_proba(test_df)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    pr_auc = float(average_precision_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred)

    return {
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "test_pr_auc": pr_auc,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": cm,
    }


def compute_threshold_table(y_test: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """
    Computes precision, recall, and F1 across custom classification thresholds.

    Why this matters for viva:
    The default 0.5 threshold can be adjusted depending on business objectives.
    If a job board wants to catch almost all scams (high recall), they might lower the threshold
    to 0.3 or 0.4 at the cost of more false positives.

    Args:
        y_test (np.ndarray): True binary test labels.
        y_prob (np.ndarray): Predicted probability of fraud.

    Returns:
        pd.DataFrame: Table with threshold trade-offs.
    """
    records: List[Dict[str, Any]] = []

    for th in THRESHOLDS:
        y_th = (y_prob >= th).astype(int)
        p = precision_score(y_test, y_th, pos_label=1, zero_division=0)
        r = recall_score(y_test, y_th, pos_label=1, zero_division=0)
        f = f1_score(y_test, y_th, pos_label=1, zero_division=0)

        records.append({
            "Threshold": th,
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1-Score": round(f, 4),
        })

    return pd.DataFrame(records)


def extract_top_logreg_words(
    logreg_pipeline: Pipeline,
    n_top: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts the top words pushing toward fraud (positive coefficients) and
    genuine (negative coefficients) from the fitted Logistic Regression pipeline.

    Args:
        logreg_pipeline (Pipeline): Fitted Logistic Regression pipeline.
        n_top (int): Number of top words to extract per class.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (top_fraud_words_df, top_genuine_words_df)
    """
    preprocessor: ColumnTransformer = logreg_pipeline.named_steps["preprocessor"]
    classifier: LogisticRegression = logreg_pipeline.named_steps["classifier"]

    # Retrieve all feature names from ColumnTransformer
    all_feature_names: np.ndarray = preprocessor.get_feature_names_out()
    coefficients: np.ndarray = classifier.coef_[0]

    # Filter only features originating from the TF-IDF text transformer
    text_mask = [f.startswith("text__") for f in all_feature_names]
    words = [f.replace("text__", "") for f, m in zip(all_feature_names, text_mask) if m]
    word_coefs = [c for c, m in zip(coefficients, text_mask) if m]

    coef_df = pd.DataFrame({"word": words, "coefficient": word_coefs})

    top_fraud = (
        coef_df.sort_values(by="coefficient", ascending=False)
        .head(n_top)
        .reset_index(drop=True)
    )
    top_genuine = (
        coef_df.sort_values(by="coefficient", ascending=True)
        .head(n_top)
        .reset_index(drop=True)
    )

    return top_fraud, top_genuine


def generate_evaluation_plots(
    model_comparison_df: pd.DataFrame,
    best_name: str,
    y_test: np.ndarray,
    best_results: Dict[str, Any],
    logreg_pipeline: Pipeline,
    reports_dir: Path,
) -> None:
    """
    Generates and saves the 4 required analytical visualizations into reports/.

    Plots generated:
    1. model_comparison.png: Grouped bar chart comparing F1 and Recall across models.
    2. confusion_matrix_best.png: Heatmap of the best model's confusion matrix.
    3. pr_curve_best.png: Precision-Recall curve with no-skill baseline and PR-AUC.
    4. top_words_logreg.png: Dual horizontal bar chart of top fraud vs genuine words.
    """
    print("\nGenerating report visualizations in reports/...")

    # -------------------------------------------------------------------------
    # Plot 1: Model Comparison Bar Chart (F1 and Recall on Test Set)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    x = np.arange(len(model_comparison_df))
    width = 0.35

    f1_vals = model_comparison_df["test_f1"].values
    rec_vals = model_comparison_df["test_recall"].values
    names = model_comparison_df["model_name"].values

    bars1 = plt.bar(x - width / 2, f1_vals, width, label="Test F1-Score", color="#2b5c8f")
    bars2 = plt.bar(x + width / 2, rec_vals, width, label="Test Recall", color="#e76f51")

    plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.title("Model Comparison on Test Set: F1-Score vs. Recall", fontsize=13, fontweight="bold")
    plt.xticks(x, names, fontsize=10)
    plt.ylim(0, 1.05)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate numeric values on bars
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        plt.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plot1_path = reports_dir / "model_comparison.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path.name}")

    # -------------------------------------------------------------------------
    # Plot 2: Confusion Matrix for Best Model
    # -------------------------------------------------------------------------
    cm = best_results["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({best_name})", fontsize=12, fontweight="bold")
    plt.colorbar(fraction=0.046, pad=0.04)

    classes = ["Genuine (0)", "Fraudulent (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    # Label text within cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            pct = (val / cm[i].sum()) * 100
            plt.text(
                j,
                i,
                f"{val:,}\n({pct:.1f}%)",
                horizontalalignment="center",
                verticalalignment="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )

    plt.ylabel("Actual Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plot2_path = reports_dir / "confusion_matrix_best.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path.name}")

    # -------------------------------------------------------------------------
    # Plot 3: Precision-Recall Curve for Best Model
    # -------------------------------------------------------------------------
    y_prob = best_results["y_prob"]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc_score = best_results["test_pr_auc"]
    no_skill = float(np.mean(y_test))

    plt.figure(figsize=(7, 5))
    plt.plot(rec, prec, color="#2a9d8f", lw=2.5, label=f"{best_name} (PR-AUC = {pr_auc_score:.4f})")
    plt.axhline(y=no_skill, color="#e63946", linestyle="--", lw=1.5, label=f"No-Skill Baseline ({no_skill:.3f})")

    plt.xlabel("Recall (Fraud Class)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Fraud Class)", fontsize=11, fontweight="bold")
    plt.title(f"Precision-Recall Curve ({best_name})", fontsize=13, fontweight="bold")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot3_path = reports_dir / "pr_curve_best.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved: {plot3_path.name}")

    # -------------------------------------------------------------------------
    # Plot 4: Top 20 Fraud and Top 20 Genuine Words from Logistic Regression
    # -------------------------------------------------------------------------
    top_fraud, top_genuine = extract_top_logreg_words(logreg_pipeline, n_top=20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Top Fraud Words (Positive coefficients)
    axes[0].barh(top_fraud["word"][::-1], top_fraud["coefficient"][::-1], color="#d62828")
    axes[0].set_title("Top 20 Words Indicating FRAUD (+ Coef)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Logistic Regression Coefficient", fontsize=10, fontweight="bold")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    # Top Genuine Words (Negative coefficients)
    # We display absolute magnitude or signed negative coefficient
    axes[1].barh(top_genuine["word"][::-1], top_genuine["coefficient"][::-1], color="#1d3557")
    axes[1].set_title("Top 20 Words Indicating GENUINE (- Coef)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Logistic Regression Coefficient", fontsize=10, fontweight="bold")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    plt.suptitle("Logistic Regression Interpretability: Influential Vocabulary Signals", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot4_path = reports_dir / "top_words_logreg.png"
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"Saved: {plot4_path.name}")


def train_and_evaluate_all() -> None:
    """
    Orchestrates the entire training, cross-validation, testing, and serialization pipeline.
    """
    train_path, test_path, models_dir, reports_dir = get_project_paths()
    train_df, test_df = load_processed_data()

    y_train = train_df["fraudulent"].values
    y_test = test_df["fraudulent"].values

    # Print baseline dummy model accuracy
    dummy_acc = float((y_test == 0).mean() * 100)
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)
    print(f"Test Set Size: {len(test_df):,} | Legitimate: {(y_test == 0).sum():,} | Fraudulent: {(y_test == 1).sum():,}")
    print(f"Dummy Classifier Accuracy (always predicts 'Genuine'): {dummy_acc:.2f}%")
    print("Why this matters for viva: Because 95.52% of test postings are genuine, a model "
          "that predicts zero fraud achieves ~95.5% accuracy! Accuracy is useless here.")
    print("=" * 70)

    # Build model candidate pipelines
    pipelines = build_model_pipelines()

    # 5-Fold Stratified Cross-Validation on the training set
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_summary_records: List[Dict[str, Any]] = []
    test_results_dict: Dict[str, Dict[str, Any]] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}

    print("\nStarting 5-Fold Stratified Cross-Validation on TRAIN set only...")
    print("-" * 70)

    for name, pipeline in pipelines.items():
        print(f"Running Cross-Validation for: {name}...")
        cv_metrics = run_cross_validation(name, pipeline, train_df, cv_splitter)
        print(f"  -> CV Fraud F1: {cv_metrics['cv_f1_mean']:.4f} (+/- {cv_metrics['cv_f1_std']:.4f}) | "
              f"Recall: {cv_metrics['cv_recall_mean']:.4f} | "
              f"Precision: {cv_metrics['cv_precision_mean']:.4f} | "
              f"PR-AUC: {cv_metrics['cv_pr_auc_mean']:.4f}")

        # Final evaluation on the unseen test set
        print(f"Evaluating {name} on unseen test set...")
        test_eval = evaluate_on_test_set(pipeline, train_df, test_df)
        fitted_pipelines[name] = pipeline
        test_results_dict[name] = test_eval

        record = {
            "model_name": name,
            **cv_metrics,
            "test_accuracy": round(test_eval["test_accuracy"], 4),
            "test_precision": round(test_eval["test_precision"], 4),
            "test_recall": round(test_eval["test_recall"], 4),
            "test_f1": round(test_eval["test_f1"], 4),
            "test_pr_auc": round(test_eval["test_pr_auc"], 4),
        }
        cv_summary_records.append(record)
        print("-" * 70)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(cv_summary_records)

    # -------------------------------------------------------------------------
    # Model Selection: Best model chosen by highest CROSS-VALIDATION F1 on TRAIN
    # -------------------------------------------------------------------------
    best_row = comparison_df.sort_values(by="cv_f1_mean", ascending=False).iloc[0]
    best_name: str = str(best_row["model_name"])
    print("\n" + "=" * 70)
    print(f"MODEL SELECTION WINNER: {best_name.upper()}")
    print(f"Selected based on 5-Fold CV Train F1-Score: {best_row['cv_f1_mean']:.4f} (+/- {best_row['cv_f1_std']:.4f})")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Print Confusion Matrices for All Models
    # -------------------------------------------------------------------------
    print("\nCONFUSION MATRICES ON TEST SET:")
    for name in pipelines.keys():
        cm = test_results_dict[name]["confusion_matrix"]
        print(f"\nModel: {name}")
        print(f"  TN (Genuine -> Genuine):     {cm[0, 0]:>5}")
        print(f"  FP (Genuine -> Fraudulent):  {cm[0, 1]:>5}")
        print(f"  FN (Fraudulent -> Genuine):  {cm[1, 0]:>5}")
        print(f"  TP (Fraudulent -> Fraud):    {cm[1, 1]:>5}")

    # -------------------------------------------------------------------------
    # Threshold Analysis for the Best Model
    # -------------------------------------------------------------------------
    best_eval = test_results_dict[best_name]
    threshold_df = compute_threshold_table(y_test, best_eval["y_prob"])
    print("\n" + "=" * 70)
    print(f"THRESHOLD TUNING TABLE ({best_name}) ON TEST SET:")
    print("=" * 70)
    print(threshold_df.to_string(index=False))
    print("Note: Keeping default threshold 0.5 for now.")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Save Artifacts
    # -------------------------------------------------------------------------
    print("\nSaving trained pipelines and metadata...")

    # 1. Save the best pipeline (refitted on entire train set)
    best_pipeline = fitted_pipelines[best_name]
    best_pipe_path = models_dir / "best_pipeline.joblib"
    joblib.dump(best_pipeline, best_pipe_path)
    print(f"Saved best pipeline: {best_pipe_path.name}")

    # 2. Save the Logistic Regression pipeline (ALWAYS saved for top words explainability)
    logreg_pipeline = fitted_pipelines["Logistic Regression"]
    logreg_pipe_path = models_dir / "logreg_pipeline.joblib"
    joblib.dump(logreg_pipeline, logreg_pipe_path)
    print(f"Saved Logistic Regression pipeline: {logreg_pipe_path.name}")

    # 3. Save model comparison CSV
    comparison_csv_path = models_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"Saved model comparison table: {comparison_csv_path.name}")

    # 4. Save metadata.json
    metadata: Dict[str, Any] = {
        "best_model_name": best_name,
        "default_threshold": 0.5,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_state": RANDOM_STATE,
        "flag_columns": FLAG_COLUMNS,
        "test_metrics": {
            "accuracy": best_eval["test_accuracy"],
            "precision": best_eval["test_precision"],
            "recall": best_eval["test_recall"],
            "f1_score": best_eval["test_f1"],
            "pr_auc": best_eval["test_pr_auc"],
        },
        "all_models_test_metrics": {
            name: {
                "accuracy": res["test_accuracy"],
                "precision": res["test_precision"],
                "recall": res["test_recall"],
                "f1_score": res["test_f1"],
                "pr_auc": res["test_pr_auc"],
            }
            for name, res in test_results_dict.items()
        },
    }
    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved training metadata: {metadata_path.name}")

    # -------------------------------------------------------------------------
    # Generate Plots
    # -------------------------------------------------------------------------
    generate_evaluation_plots(
        model_comparison_df=comparison_df,
        best_name=best_name,
        y_test=y_test,
        best_results=best_eval,
        logreg_pipeline=logreg_pipeline,
        reports_dir=reports_dir,
    )

    # -------------------------------------------------------------------------
    # Print Clean Final Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("FINAL MODEL BENCHMARK SUMMARY TABLE")
    print("=" * 90)
    summary_display = pd.DataFrame({
        "Model": comparison_df["model_name"],
        "CV F1 (Train)": comparison_df.apply(lambda r: f"{r['cv_f1_mean']:.3f} ± {r['cv_f1_std']:.3f}", axis=1),
        "CV PR-AUC": comparison_df.apply(lambda r: f"{r['cv_pr_auc_mean']:.3f} ± {r['cv_pr_auc_std']:.3f}", axis=1),
        "Test Accuracy": comparison_df["test_accuracy"].apply(lambda v: f"{v*100:.2f}%"),
        "Test Precision": comparison_df["test_precision"].apply(lambda v: f"{v:.4f}"),
        "Test Recall": comparison_df["test_recall"].apply(lambda v: f"{v:.4f}"),
        "Test F1": comparison_df["test_f1"].apply(lambda v: f"{v:.4f}"),
        "Test PR-AUC": comparison_df["test_pr_auc"].apply(lambda v: f"{v:.4f}"),
    })
    print(summary_display.to_string(index=False))
    print("=" * 90)
    print(f"Best Model Selected: {best_name}")
    print("=" * 90)


def main() -> None:
    """
    Main entry point for training and evaluation.
    """
    train_and_evaluate_all()


if __name__ == "__main__":
    main()
