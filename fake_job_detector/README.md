# Fake Job Posting Detector

An integrated academic machine learning and software engineering project developed for a **Machine Learning** course and an **Advanced Python** course.

---

## 1. Project Overview & Purpose

Online employment scams, fraudulent job advertisements, and fake internships present severe risks to student job seekers, including identity theft, financial loss, and phishing. 

The **Fake Job Posting Detector** is an explainable decision-support tool designed to screen job postings and highlight suspicious indicators before applicants share personal details or pay fraudulent fees. The system combines:
- A supervised **Machine Learning NLP classification pipeline** trained on real-world job posting data.
- A heuristic **Rule Checker** monitoring concrete red-flag patterns (e.g. fee requests, free email domains, Telegram/WhatsApp contacts).
- **Local feature explainability** via logistic regression coefficients, revealing which words pushed the risk score up or down.
- A secure, multi-page **Streamlit web application** backed by SQLite and bcrypt authentication.

> **Important Concept (Risk Score vs. Probability)**: The score produced by the application is a **risk score** (a model estimate of statistical similarity to known fraudulent postings), **never a true probability**. Because the model utilizes class weighting to counter extreme dataset imbalance, raw output values do not correspond to true population probabilities, nor do they constitute definitive legal proof.

---

## 2. Features

- **User Authentication (Login & Register)**: User accounts with password hashing (bcrypt), input validation, session persistence, and role-based access control (standard user vs. administrator).
- **Check a Job Ad**: Interactive evaluation form accepting job title, company profile, description body, requirements, benefits, salary range, and posting flags (telecommuting, company logo, screening questions).
- **3-Tier Calibrated Risk Scoring**:
  - **LOW Risk (< 40%)** &rarr; Classification: `"Likely genuine"`
  - **MEDIUM Risk (40% to < 50%)** &rarr; Classification: `"Borderline - review carefully"`
  - **HIGH Risk (&ge; 50%)** &rarr; Classification: `"Potentially fraudulent"`
- **Rule-Based Warning Signs**: Independent heuristic rules detecting 7 high-risk patterns: missing company profile, upfront fee requests, unrealistic earnings, urgent recruitment language, free email / messaging contacts, unusually short descriptions, and "no experience required" hooks.
- **Highlighted Suspicious Phrases**: Interactive visual inspection rendering detected suspicious phrases wrapped in highlighted `<mark>` tags, safely escaping all HTML/script tags.
- **Explainable Influential Words**: Word-level contribution breakdown showing top 5 words pushing toward fraud and top 5 words pushing toward genuine postings based on model coefficients.
- **My History & Audit Log**: Searchable history of past checks for logged-in users, with filtering by risk level, detailed view, deletion capability, and PDF re-download.
- **Compare Ads**: Side-by-side comparative inspection tool evaluating two job advertisements simultaneously across metrics, warnings, and influential words without saving comparisons to the database.
- **Downloadable PDF Audit Reports**: Programmatic, in-memory PDF generation (ReportLab) containing complete audit breakdowns, metadata, warning indicators, and influential words.
- **Model Insights Dashboard**: Transparent inspection page showcasing model comparisons, confusion matrices, precision-recall curves, top vocabulary coefficients, and dataset limitations.
- **Admin Dashboard**: Secure administrative console with platform-wide inspection volume statistics, risk distribution analytics, full user directory, and complete audit log access.

---

## 3. Tech Stack

- **Language**: Python 3.12
- **Web Interface**: Streamlit (multi-page application architecture)
- **Data Manipulation**: pandas, NumPy
- **Machine Learning**: scikit-learn (Pipelines, ColumnTransformer, TF-IDF, Logistic Regression, Naive Bayes, Random Forest)
- **Model Persistence**: joblib
- **Visualizations**: Matplotlib, Altair
- **Database**: SQLite3 (built-in relational database)
- **Security & Cryptography**: bcrypt (salted password hashing)
- **Document Generation**: ReportLab (in-memory PDF generation)

---

## 4. Folder Structure

```text
fake_job_detector/
├── .venv/                               # Python virtual environment (ignored by version control)
├── data/
│   ├── raw/
│   │   ├── .gitkeep                     # Preserves empty raw data directory in repository
│   │   └── fake_job_postings.csv        # Kaggle EMSCAD raw CSV dataset (download required)
│   ├── processed/
│   │   ├── .gitkeep                     # Preserves processed directory in repository
│   │   ├── train.csv                    # Cleaned, deduplicated training split (80% stratified)
│   │   └── test.csv                     # Cleaned, deduplicated test split (20% stratified)
│   └── app.db                           # SQLite database storing users and historical check logs
├── models/
│   ├── .gitkeep                         # Preserves models directory in repository
│   ├── best_pipeline.joblib             # Serialized production pipeline (Logistic Regression)
│   ├── logreg_pipeline.joblib           # Serialized Logistic Regression pipeline for local explainability
│   ├── metadata.json                    # Serialized model metrics, parameters, and risk cutoffs
│   └── model_comparison.csv             # Cross-validation and test set evaluation comparison table
├── reports/
│   ├── .gitkeep                         # Preserves reports directory in repository
│   ├── confusion_matrix_best.png        # Test set confusion matrix visualization
│   ├── model_comparison.png             # Performance comparison bar chart across models
│   ├── pr_curve_best.png                # Precision-Recall curve visualization with baseline
│   └── top_words_logreg.png             # Feature importance chart of top model coefficients
├── app.py                               # Streamlit multi-page web application interface
├── auth.py                              # User authentication, bcrypt password hashing, and session management
├── calibrate_risk_levels.py             # Script determining data-driven LOW, MEDIUM, HIGH risk cutoffs
├── database.py                          # SQLite database schema, connection handling, and CRUD operations
├── predictor.py                         # Inference engine, feature engineering, and explainability
├── prepare_data.py                      # Data cleaning, deduplication, flag engineering, and stratified split
├── README.md                            # Comprehensive project documentation and setup guide
├── reports.py                           # PDF audit report generator built using ReportLab
├── requirements.txt                     # Third-party Python dependencies
├── rules.py                             # Rule-based heuristic red-flag checker and HTML text highlighter
├── smoke_test.py                        # Sanity test suite verifying environment, files, and end-to-end inference
├── test_label.py                        # Unit and integration test verifying 3-tier risk classification labels
└── train_model.py                       # Model training, 5-fold cross-validation, evaluation, and chart generation
```

---

## 5. Setup on Windows PowerShell

Open PowerShell and navigate to the `fake_job_detector` folder:

```powershell
# 1. Navigate to the project directory
cd "c:\Users\Abhishek\OneDrive\Desktop\fake job posting detector\fake_job_detector"

# 2. Create Python virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip and install all project dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Run Order & Pipeline Execution

Execute the pipeline in the following sequence from your activated virtual environment:

### Step 1: Download Raw Dataset
Download `fake_job_postings.csv` from Kaggle (see link in Section 11) and place it directly into `data/raw/fake_job_postings.csv`.

### Step 2: Clean & Split Data
```powershell
python prepare_data.py
```
Cleans raw text (HTML unescaping, lowercasing, punctuation stripping), flags missing company profiles and salary ranges, removes duplicate postings across subsets to prevent data leakage, and outputs stratified splits into `data/processed/train.csv` and `data/processed/test.csv`.

### Step 3: Train & Compare Models
```powershell
python train_model.py
```
Trains Logistic Regression, Multinomial Naive Bayes, and Random Forest using 5-fold Stratified Cross-Validation on the training split. Evaluates performance on the held-out test split, generates charts in `reports/`, and serializes the best model into `models/best_pipeline.joblib` and `models/metadata.json`.

### Step 4: Calibrate Risk Levels
```powershell
python calibrate_risk_levels.py
```
Evaluates out-of-fold cross-validation predictions on the training data to select empirical risk score cutoffs:
- Low risk threshold (`low_max = 0.40`) calibrated to guarantee &ge; 90% fraud recall (achieves ~93.4%).
- High risk threshold (`high_min = 0.50`) calibrated where precision reaches &ge; 50%.
Saves calibrated cutoffs into `models/metadata.json`.

### Step 5: Create Administrator Account
```powershell
python auth.py create-admin
```
Prompts interactively for a username, email, and password to initialize an administrator account in `data/app.db`. Passwords are encrypted with bcrypt using per-user unique salt rounds.

### Step 6: Launch Streamlit Web Application
```powershell
streamlit run app.py
```
Launches the web application locally (default: `http://localhost:8501`).

---

## 7. How the Machine Learning Works (In Plain Words)

1. **Text Representations (TF-IDF)**:
   The job title, company profile, description, requirements, and benefits are consolidated into one cleaned text corpus. A TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer transforms text into 10,000 unigram and bigram numerical features. Common, generic English stop words are removed, while terms unique to scams (e.g. "wire transfer", "registration fee", "data entry") receive higher weight.

2. **Engineered Structural Flags**:
   Scam postings frequently exhibit distinct structural patterns independent of vocabulary. Six binary features are integrated alongside TF-IDF:
   - `telecommuting`: Whether the position is remote.
   - `has_company_logo`: Whether an official employer logo was uploaded.
   - `has_questions`: Whether screening questions were configured.
   - `missing_company_profile`: 1 if the employer profile is absent.
   - `missing_salary`: 1 if no salary interval is specified.
   - `short_description`: 1 if the job description contains fewer than 30 words.

3. **Handling Class Imbalance (Class Weighting)**:
   Only ~4.5% of job ads in the dataset are fraudulent (~95.5% genuine). If a standard classifier were trained without intervention, it could simply guess "genuine" for every ad and claim **95.5% accuracy** while missing 100% of scams! To prevent this, we:
   - Use `class_weight="balanced"` in Logistic Regression to penalize false negatives severely.
   - Avoid plain SMOTE over sparse TF-IDF feature spaces, as synthetic interpolation creates unrealistic high-dimensional word combinations.
   - Employ Stratified K-Fold splitting to preserve the 4.5% fraud ratio across all folds.

4. **Model Selection Rationale (Logistic Regression vs. Random Forest)**:
   The production model was selected by mean cross-validated F1-score on the training split:
   - **Logistic Regression**: CV F1 of **0.6439** (Recall: **89.89%**, Precision: **50.19%**).
   - **Random Forest**: CV F1 of **0.6333** (Recall: **47.39%**, Precision: **96.09%**).
   While Random Forest achieved higher precision, it missed more than half of all fraudulent ads. In a scam-screening tool designed to safeguard students, **high recall** is critical so that fraudulent postings do not slip through undetected. Furthermore, Logistic Regression provides direct, transparent feature coefficients for local explainability.

5. **Data-Driven Risk Cutoffs**:
   Rather than applying arbitrary thresholds, the cutoffs were calibrated on out-of-fold training predictions:
   - **LOW (< 0.40)**: Captures legitimate postings with high confidence (&ge; 93% fraud recall). Output: `"Likely genuine"`.
   - **MEDIUM (0.40 to < 0.50)**: Postings with ambiguous characteristics (missing company profiles or brief descriptions). Output: `"Borderline - review carefully"`.
   - **HIGH (&ge; 0.50)**: Postings exhibiting strong statistical alignment with scam patterns where precision reaches &ge; 50%. Output: `"Potentially fraudulent"`.

6. **Separate Heuristic Rule Layer**:
   The 7 warning signs in `rules.py` operate completely independently of the statistical model. They look for explicit scam triggers (e.g., upfront payment demands, WhatsApp contacts) to provide defense-in-depth, ensuring that even if an ad uses clever wording that fools the ML model, explicit red flags are still caught.

---

## 8. Results & Performance Comparison

The following performance figures are extracted directly from `models/metadata.json` and `models/model_comparison.csv` evaluated on the Kaggle EMSCAD dataset:

### 5-Fold Stratified Cross-Validation on Training Set (Selection Metric)

| Model | CV F1-Score | CV Precision | CV Recall | CV PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **0.6439 &plusmn; 0.0229** | 0.5019 &plusmn; 0.0248 | **0.8989 &plusmn; 0.0180** | 0.8326 &plusmn; 0.0290 |
| Random Forest | 0.6333 &plusmn; 0.0499 | **0.9609 &plusmn; 0.0343** | 0.4739 &plusmn; 0.0498 | **0.8403 &plusmn; 0.0298** |
| Multinomial Naive Bayes | 0.2721 &plusmn; 0.0312 | 0.4202 &plusmn; 0.0506 | 0.2022 &plusmn; 0.0261 | 0.3110 &plusmn; 0.0288 |

### Unseen Test Set Evaluation (Held-Out Split at 0.5 Default Cutoff)

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **0.9540** | 0.4921 | **0.8611** | 0.6263 | 0.8007 |
| Random Forest | **0.9764** | **0.9595** | 0.4931 | **0.6514** | **0.8183** |
| Multinomial Naive Bayes | 0.9540 | 0.4688 | 0.2083 | 0.2885 | 0.3540 |

> **Viva Note on Test Metrics**: On the held-out test set, Logistic Regression achieves **86.11% recall** (detecting 124 out of 144 real scams in the test set). Precision at the 0.5 cutoff is **49.21%**, meaning approximately 1 in 2 ads flagged as High Risk are confirmed scams—a favorable tradeoff for a protective screening system.

---

## 9. Security Notes

- **Password Storage**: Passwords are never stored in plaintext. `auth.py` hashes passwords using `bcrypt` with individual salt rounds.
- **SQL Injection Prevention**: All SQLite queries in `database.py` use parameter binding (`?` placeholders). No user inputs are concatenated directly into SQL queries.
- **Cross-Site Scripting (XSS) Mitigation**: User-submitted text displayed in HTML markup (such as the phrase highlighting widget in `rules.py`) is sanitized via `html.escape()` prior to formatting.
- **PDF Generation Defense**: Text rendered in ReportLab PDF tables is sanitized against XML entities (`xml.sax.saxutils.escape`) and stripped of non-Latin characters to prevent document compilation errors.
- **Server-Side Authorization**: Administrative controls (the Admin Dashboard) verify `is_admin == 1` server-side from session state; navigation options are hidden from non-admin users.
- **Data Isolation**: Logged-in users can only inspect, filter, or delete their own historical checks.
- **Administrative Visibility**: Administrators have read-only access to view all submitted checks across users for auditing purposes.
- **No Default Credentials**: The system ships without hardcoded passwords or default administrative credentials. All accounts must be explicitly created via the registration form or `python auth.py create-admin`.

---

## 10. Limitations

- **Historical Dataset (2012–2014)**: The EMSCAD dataset reflects employment ads collected between 2012 and 2014. Modern scam tactics (e.g. AI-generated descriptions, cryptocurrency payroll schemes, advanced LinkedIn phishing) may exhibit vocabulary shifts.
- **Single Source Bias**: Data originates from one primary job board aggregator; language styles may not represent specialized freelance, gig, or non-English job boards.
- **Language Scope**: Text processing and TF-IDF pipelines are designed exclusively for English-language advertisements.
- **Risk Score is Not Probability**: Output scores reflect statistical distance and similarity to weighted scam clusters, not a mathematical certainty or population prevalence.
- **Two-Way Classification Error**: The model can produce false positives (legitimate startup ads with minimal details flagged as high risk) and false negatives (sophisticated fraud using polished corporate jargon).
- **Rule Independence**: Heuristic warning signs operate independently from model coefficients; an ad may score low on text modeling while triggering rule-based warnings, or vice-versa.

---

## 11. Dataset Credit & Reference

- **Dataset**: Real / Fake Job Posting Prediction (EMSCAD - Employment Scam Aegean Dataset)
- **Author / Source**: University of the Aegean, Laboratory of Information & Communication Systems Security (ICBSS)
- **Kaggle URL**: [Real / Fake Job Posting Prediction on Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction/data)
- **Original Citation**: Vidros, S., Kolias, C., Kambourakis, G., & Akoglu, L. (2017). *Automatic Detection of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset*. Future Internet, 9(1), 6.
