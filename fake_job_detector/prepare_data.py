"""
Data Preparation & Cleaning Pipeline for Fake Job Posting Detector.

This module prepares raw job posting data for machine learning model training:
1. Loads raw CSV data without guessing column names or data types.
2. Deduplicates repeated postings BEFORE splitting to avoid train-test data leakage.
3. Engineers heuristic missingness and brevity flag features from the raw data.
4. Aggregates and cleans text fields (HTML entity unescaping, tag stripping, normalization).
5. Splits the data into 80/20 train and test sets with stratification on the target class.
6. Saves processed datasets ready for TF-IDF vectorization and classification.

Usage:
    python prepare_data.py            # Runs the full cleaning and preparation pipeline
    python prepare_data.py --inspect  # Runs diagnostic inspection on raw data only
"""

import argparse
import html
from pathlib import Path
import re
from typing import List, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

# Fixed random seed for reproducibility across all data operations and splits
RANDOM_STATE: int = 42


def get_raw_data_path() -> Path:
    """
    Locates the raw CSV file inside the project directory.

    Why this matters:
    Using pathlib relative path resolution ensures the script runs smoothly whether
    executed from inside `fake_job_detector/` or from the parent workspace folder.

    Returns:
        Path: Path pointing to the raw dataset CSV file.

    Raises:
        FileNotFoundError: If 'fake_job_postings.csv' cannot be found.
    """
    script_dir: Path = Path(__file__).resolve().parent
    candidates: List[Path] = [
        script_dir / "data" / "raw" / "fake_job_postings.csv",
        Path("data/raw/fake_job_postings.csv"),
        Path("fake_job_detector/data/raw/fake_job_postings.csv"),
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find 'fake_job_postings.csv' in expected data/raw locations. "
        "Please ensure the CSV file is placed in fake_job_detector/data/raw/."
    )


def clean_text_content(raw_text: str) -> str:
    """
    Cleans and standardizes raw text from job postings.

    Steps applied:
    1. html.unescape(): Converts HTML entities like '&amp;' -> '&', '&lt;' -> '<'.
       Why: Prevents tokenizers from interpreting '&amp;' and '&' as different tokens.
    2. re.sub(r'<[^>]+>', ' '): Removes raw HTML markup tags (e.g., <p>, <br/>, <div>).
       Why: Formatting tags are noise and do not carry semantic job description content.
    3. lower(): Lowercases all characters.
       Why: Ensures case-insensitive vocabulary matching (e.g., 'Python' == 'python').
    4. re.sub(r'\\s+', ' '): Collapses multiple tabs/newlines/spaces into a single space.
       Why: Compacts text and standardizes word boundaries.

    Args:
        raw_text (str): Raw string containing job posting text.

    Returns:
        str: Cleaned, normalized, single-spaced lowercase text.
    """
    # Decode HTML entities
    text: str = html.unescape(raw_text)

    # Strip HTML tags by replacing them with a space (avoids concatenating adjacent words)
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize to lowercase for consistent vocabulary indexing
    text = text.lower()

    # Collapse multiple whitespaces and strip leading/trailing spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_and_clean_data(raw_csv_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the complete data cleaning, feature engineering, and splitting pipeline.

    Order of Operations:
    1. Load raw dataset.
    2. Check and remove content duplicates BEFORE splitting (prevents train-test leakage).
    3. Create binary flag columns from raw data BEFORE filling missing values
       (crucial for viva: once NaN values are filled with empty strings, the missingness
       signal is lost!).
    4. Construct a unified 'text' column, cleaning HTML and whitespace.
    5. Perform 80/20 train/test stratified split using fixed RANDOM_STATE = 42.
    6. Export train.csv and test.csv with only the 10 selected modeling columns.
    7. Print a concise summary report.

    Args:
        raw_csv_path (Optional[Path]): Optional explicit path to raw CSV.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The final (train_df, test_df) DataFrames.
    """
    # -------------------------------------------------------------------------
    # Step 1: Load raw dataset
    # -------------------------------------------------------------------------
    csv_file: Path = raw_csv_path if raw_csv_path is not None else get_raw_data_path()
    print(f"Loading raw dataset from: {csv_file.name}")
    df: pd.DataFrame = pd.read_csv(csv_file)
    rows_before: int = len(df)
    fraud_counts_before: dict = df["fraudulent"].value_counts().to_dict()

    # -------------------------------------------------------------------------
    # Step 2: Remove content duplicates BEFORE splitting
    # Why this matters for viva:
    # If the exact same job ad appears multiple times, having one copy in train
    # and another in test causes "data leakage". The model would simply memorize
    # the leaked ad, producing overly optimistic test metrics that fail on unseen data.
    # -------------------------------------------------------------------------
    dup_columns: List[str] = [
        "title", "company_profile", "description", "requirements", "benefits"
    ]

    # Check for any duplicate groups with conflicting labels (e.g., same ad marked both 0 and 1)
    # Why: Conflicting labels indicate labeling noise or poisoned data that could confuse the model.
    conflicting_groups = (
        df.groupby(dup_columns)["fraudulent"]
        .nunique()
        .loc[lambda count: count > 1]
    )
    num_conflicting_groups: int = len(conflicting_groups)
    print(f"Duplicate groups with conflicting labels: {num_conflicting_groups}")

    # Deduplicate keeping the first occurrence
    df_dedup: pd.DataFrame = df.drop_duplicates(subset=dup_columns, keep="first").copy()
    rows_after: int = len(df_dedup)
    fraud_counts_after: dict = df_dedup["fraudulent"].value_counts().to_dict()

    # -------------------------------------------------------------------------
    # Step 3: Create flag columns from RAW data (BEFORE filling missing values)
    # Why this matters for viva:
    # Feature engineering missingness flags MUST happen on the raw data.
    # If we replace NaN with "" first to prepare the text column, we would erase
    # the information that the company profile or salary was missing!
    # -------------------------------------------------------------------------
    # missing_company_profile: 1 if company_profile was NaN/empty, else 0
    # Recall from inspection: missing company profile had a ~17.7% fraud rate vs ~1.9% present!
    df_dedup["missing_company_profile"] = df_dedup["company_profile"].isna().astype(int)

    # missing_salary: 1 if salary_range was NaN/empty, else 0
    df_dedup["missing_salary"] = df_dedup["salary_range"].isna().astype(int)

    # short_description: 1 if description has fewer than 30 words, else 0
    # Why: Scam postings frequently post curt, vague listings with minimal job scope details.
    df_dedup["short_description"] = df_dedup["description"].apply(
        lambda desc: 1 if pd.isna(desc) or len(str(desc).split()) < 30 else 0
    )

    # -------------------------------------------------------------------------
    # Step 4: Build one combined text column called 'text'
    # Why: A single consolidated text column allows TF-IDF vectorizers to capture
    # vocabulary patterns across title, requirements, company profile, and benefits.
    # -------------------------------------------------------------------------
    # Fill missing values with empty string solely for text concatenation
    text_components = df_dedup[dup_columns].fillna("").agg(" ".join, axis=1)
    df_dedup["text"] = text_components.apply(clean_text_content)

    # -------------------------------------------------------------------------
    # Step 5: Split into train and test (80/20, stratified on 'fraudulent')
    # Why this matters for viva:
    # Because fraudulent ads are rare (~4.5%), random splitting could accidentally
    # place disproportionately few or many fraud cases in the test set. Stratified
    # splitting guarantees that train and test preserve the exact same class ratio.
    # -------------------------------------------------------------------------
    train_df, test_df = train_test_split(
        df_dedup,
        test_size=0.20,
        stratify=df_dedup["fraudulent"],
        random_state=RANDOM_STATE
    )

    # -------------------------------------------------------------------------
    # Step 6: Save selected columns to processed CSV files
    # -------------------------------------------------------------------------
    selected_columns: List[str] = [
        "job_id",
        "title",
        "text",
        "telecommuting",
        "has_company_logo",
        "has_questions",
        "missing_company_profile",
        "missing_salary",
        "short_description",
        "fraudulent"
    ]

    train_export: pd.DataFrame = train_df[selected_columns].copy()
    test_export: pd.DataFrame = test_df[selected_columns].copy()

    # Determine output paths relative to script location
    script_dir: Path = Path(__file__).resolve().parent
    processed_dir: Path = script_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path: Path = processed_dir / "train.csv"
    test_path: Path = processed_dir / "test.csv"

    train_export.to_csv(train_path, index=False)
    test_export.to_csv(test_path, index=False)

    # -------------------------------------------------------------------------
    # Step 7: Print concise summary report
    # -------------------------------------------------------------------------
    train_fraud_cnt: int = int((train_export["fraudulent"] == 1).sum())
    train_fraud_pct: float = (train_fraud_cnt / len(train_export)) * 100

    test_fraud_cnt: int = int((test_export["fraudulent"] == 1).sum())
    test_fraud_pct: float = (test_fraud_cnt / len(test_export)) * 100

    # Calculate average word count by class in the cleaned dataset
    all_word_counts: pd.Series = df_dedup["text"].apply(lambda t: len(t.split()))
    genuine_avg_words: float = float(all_word_counts[df_dedup["fraudulent"] == 0].mean())
    fraud_avg_words: float = float(all_word_counts[df_dedup["fraudulent"] == 1].mean())

    sample_cleaned_text: str = str(df_dedup["text"].iloc[0])[:300]

    print("\n" + "=" * 60)
    print("DATA PREPARATION PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Rows before deduplication:      {rows_before:,} (Fraud: {fraud_counts_before.get(1, 0)}, Genuine: {fraud_counts_before.get(0, 0)})")
    print(f"Rows after deduplication:       {rows_after:,} (Fraud: {fraud_counts_after.get(1, 0)}, Genuine: {fraud_counts_after.get(0, 0)})")
    print(f"Removed duplicates:             {rows_before - rows_after:,}")
    print("-" * 60)
    print(f"Train set size (80%):           {len(train_export):,} rows")
    print(f"  - Fraudulent postings:        {train_fraud_cnt} ({train_fraud_pct:.2f}%)")
    print(f"Test set size (20%):            {len(test_export):,} rows")
    print(f"  - Fraudulent postings:        {test_fraud_cnt} ({test_fraud_pct:.2f}%)")
    print("-" * 60)
    print("Average text length (in words):")
    print(f"  - Genuine postings:           {genuine_avg_words:.1f} words")
    print(f"  - Fraudulent postings:        {fraud_avg_words:.1f} words")
    print("-" * 60)
    print(f"Sample cleaned text preview (first 300 chars):")
    print(f"\"{sample_cleaned_text}...\"")
    print("-" * 60)
    print(f"Saved processed train set to:   {train_path}")
    print(f"Saved processed test set to:    {test_path}")
    print("=" * 60 + "\n")

    return train_export, test_export


def inspect_raw_dataset(csv_path: Optional[Path] = None) -> None:
    """
    Inspects the raw job posting dataset and prints diagnostic insights.

    This function does NOT modify, clean, or split the data. It performs 8 key diagnostic
    checks to guide subsequent feature engineering, cleaning, and model selection.

    Args:
        csv_path (Optional[Path]): Path to the CSV file. If None, resolves automatically.
    """
    resolved_path: Path = csv_path if csv_path is not None else get_raw_data_path()
    print("=" * 80)
    print(f"INSPECTING RAW DATASET: {resolved_path.name}")
    print("=" * 80)

    df: pd.DataFrame = pd.read_csv(resolved_path)

    # 1. Dimensions
    num_rows: int = df.shape[0]
    num_cols: int = df.shape[1]
    print("\n--- 1. DATASET DIMENSIONS ---")
    print(f"Total Rows (Postings):    {num_rows:,}")
    print(f"Total Columns (Features): {num_cols}")

    # 2. Columns & Data Types
    print("\n--- 2. COLUMN NAMES & DATA TYPES ---")
    dtypes_df: pd.DataFrame = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })
    print(dtypes_df.to_string(index=False))

    # 3. Missing Values
    print("\n--- 3. MISSING VALUES PER COLUMN (Sorted Descending) ---")
    missing_count: pd.Series = df.isnull().sum()
    missing_pct: pd.Series = (missing_count / num_rows) * 100
    missing_df: pd.DataFrame = pd.DataFrame({
        "Missing Count": missing_count,
        "Percentage (%)": missing_pct.round(2)
    }).sort_values(by="Missing Count", ascending=False)
    print(missing_df.to_string())

    # 4. Target Class Balance
    print("\n--- 4. TARGET CLASS BALANCE ('fraudulent') ---")
    fraud_counts: pd.Series = df["fraudulent"].value_counts(dropna=False)
    fraud_pcts: pd.Series = (df["fraudulent"].value_counts(normalize=True, dropna=False) * 100).round(2)
    balance_df: pd.DataFrame = pd.DataFrame({
        "Count": fraud_counts,
        "Percentage (%)": fraud_pcts
    })
    balance_df.index = balance_df.index.map({0: "0 (Legitimate)", 1: "1 (Fraudulent)"})
    print(balance_df.to_string())
    imbalance_ratio: float = round(fraud_counts.get(0, 0) / max(fraud_counts.get(1, 1), 1), 1)
    print(f"\nImbalance Ratio: Approximately {imbalance_ratio}:1 legitimate-to-fraudulent.")

    # 5. Duplicates
    print("\n--- 5. DUPLICATE ROWS CHECK ---")
    exact_duplicates: int = int(df.duplicated().sum())
    duplicates_excl_id: int = int(df.drop(columns=["job_id"], errors="ignore").duplicated().sum())
    print(f"Exact full-row duplicates (including job_id): {exact_duplicates}")
    print(f"Content duplicates (ignoring unique job_id):   {duplicates_excl_id}")

    # 6. Structured Columns
    print("\n--- 6. FRAUD RATES FOR STRUCTURED BINARY COLUMNS ---")
    for col in ["telecommuting", "has_company_logo", "has_questions"]:
        if col in df.columns:
            grouped = df.groupby(col)["fraudulent"].agg(
                Total_Count="count",
                Fraud_Count="sum",
                Fraud_Rate=lambda s: round(s.mean() * 100, 2)
            )
            print(f"\nFeature: '{col}'")
            print(grouped.to_string())

    # 7. Company Profile Missing vs Present
    print("\n--- 7. FRAUD RATE BY 'company_profile' PRESENCE ---")
    has_profile: pd.Series = df["company_profile"].notnull()
    profile_analysis: pd.DataFrame = df.groupby(has_profile)["fraudulent"].agg(
        Total_Count="count",
        Fraud_Count="sum",
        Fraud_Rate=lambda s: round(s.mean() * 100, 2)
    )
    profile_analysis.index = ["Missing Company Profile (NaN)", "Present Company Profile"]
    print(profile_analysis.to_string())

    # 8. Example Rows
    print("\n--- 8. EXAMPLE ROWS COMPARISON (Text Truncated to 200 Characters) ---")
    text_columns: List[str] = [
        "title", "company_profile", "description", "requirements", "benefits"
    ]
    genuine_row = df[df["fraudulent"] == 0].iloc[0]
    fraud_row = df[df["fraudulent"] == 1].iloc[0]

    def display_sample_row(row: pd.Series, label_desc: str) -> None:
        print(f"\n>>> Sample: {label_desc} (job_id: {row.get('job_id')})")
        print(f"    Location:           {row.get('location')}")
        print(f"    Has Company Logo:   {row.get('has_company_logo')}")
        print(f"    Has Questions:      {row.get('has_questions')}")
        print(f"    Telecommuting:      {row.get('telecommuting')}")
        print(f"    Employment Type:    {row.get('employment_type')}")
        print(f"    Industry:           {row.get('industry')}")
        for col in text_columns:
            val = str(row.get(col, ""))
            if val == "nan" or not val.strip():
                truncated = "[MISSING / EMPTY]"
            else:
                clean_text = " ".join(val.split())
                truncated = clean_text[:200] + ("..." if len(clean_text) > 200 else "")
            print(f"    {col.capitalize()}: {truncated}")

    display_sample_row(genuine_row, "GENUINE JOB POSTING (Label = 0)")
    display_sample_row(fraud_row, "FRAUDULENT JOB POSTING (Label = 1)")
    print("=" * 80)


def main() -> None:
    """
    Main entry point with command-line argument parsing.

    Default:
        python prepare_data.py            -> Runs data cleaning and exports train/test sets.
    Inspection:
        python prepare_data.py --inspect  -> Runs diagnostic inspection on raw data.
    """
    parser = argparse.ArgumentParser(
        description="Data Preparation & Cleaning Pipeline for Fake Job Posting Detector."
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Run diagnostic inspection of the raw dataset instead of the cleaning pipeline."
    )

    args = parser.parse_args()

    if args.inspect:
        inspect_raw_dataset()
    else:
        prepare_and_clean_data()


if __name__ == "__main__":
    main()
