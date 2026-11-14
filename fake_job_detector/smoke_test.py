"""
End-to-End Smoke Test Suite for Fake Job Posting Detector.

Validates core system integrity before demonstration or viva examination:
1. Verifies existence of all code, model, dataset, and artifact files.
2. Confirms clean module imports across all project files without side-effects.
3. Evaluates Predictor inference on genuine and fraudulent postings (verifying score ordering and labels).
4. Asserts data-driven risk cutoffs in metadata.json satisfy 0 < low_max < high_min < 1.
5. Verifies label_for_level() returns exact 3-tier strings across risk levels.
6. Executes self-test suites of database.py, rules.py, and reports.py in isolation.
7. Executes test_label.py automated test suite.

Note:
This test suite does NOT modify or write to data/app.db.
"""

import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import List, Tuple

# Project root directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str) -> None:
    """Prints a styled section header."""
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def print_check(name: str, passed: bool, detail: str = "") -> None:
    """Formats and prints an individual check result."""
    status = "PASS" if passed else "FAIL"
    detail_str = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{detail_str}")


def check_required_files() -> bool:
    """
    Check 1: Asserts that all essential source code files, pre-trained pipelines,
    metadata configurations, and processed dataset splits exist.
    """
    required_paths: List[Path] = [
        PROJECT_ROOT / "prepare_data.py",
        PROJECT_ROOT / "train_model.py",
        PROJECT_ROOT / "calibrate_risk_levels.py",
        PROJECT_ROOT / "predictor.py",
        PROJECT_ROOT / "rules.py",
        PROJECT_ROOT / "database.py",
        PROJECT_ROOT / "auth.py",
        PROJECT_ROOT / "reports.py",
        PROJECT_ROOT / "test_label.py",
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "models" / "best_pipeline.joblib",
        PROJECT_ROOT / "models" / "logreg_pipeline.joblib",
        PROJECT_ROOT / "models" / "metadata.json",
        PROJECT_ROOT / "models" / "model_comparison.csv",
        PROJECT_ROOT / "data" / "processed" / "train.csv",
        PROJECT_ROOT / "data" / "processed" / "test.csv",
    ]

    all_exist = True
    missing_files: List[str] = []

    for path in required_paths:
        if not path.is_file():
            all_exist = False
            missing_files.append(str(path.relative_to(PROJECT_ROOT)))

    if all_exist:
        print_check("Check 1: Required files existence", True, f"All {len(required_paths)} core files present.")
    else:
        print_check("Check 1: Required files existence", False, f"Missing {len(missing_files)} file(s): {', '.join(missing_files)}")

    return all_exist


def check_module_imports() -> bool:
    """
    Check 2: Imports all project modules dynamically to verify syntax and dependency resolution.
    Does not initialize any state that writes to data/app.db.
    """
    modules_to_test = [
        "auth",
        "database",
        "predictor",
        "reports",
        "rules",
        "prepare_data",
        "calibrate_risk_levels",
        "train_model",
        "app",
    ]

    all_imported = True
    failed_modules: List[Tuple[str, str]] = []

    for mod_name in modules_to_test:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:
            all_imported = False
            failed_modules.append((mod_name, str(exc)))

    if all_imported:
        print_check("Check 2: Module imports", True, f"Successfully imported {len(modules_to_test)} project modules.")
    else:
        err_msg = "; ".join(f"{m}: {e}" for m, e in failed_modules)
        print_check("Check 2: Module imports", False, f"Import failures: {err_msg}")

    return all_imported


def check_predictor_inference() -> bool:
    """
    Check 3: Evaluates Predictor inference on a genuine-style ad and a scam-style ad.
    Verifies that the scam ad produces a strictly higher risk score than the genuine ad,
    and that classification labels exactly match the risk levels.
    """
    try:
        from predictor import Predictor, label_for_level
        predictor = Predictor()

        # Genuine-style ad: Realistic software internship with company profile
        genuine_result = predictor.predict(
            title="Software Engineering Intern",
            company_profile=(
                "TechCorp is a leading enterprise cloud software provider founded in 2012 "
                "specializing in developer productivity tools, APIs, and scalable infrastructure."
            ),
            description=(
                "We are seeking a motivated Software Engineering Intern to join our core backend "
                "engineering team. The intern will build microservices in Python, participate in "
                "code reviews, write unit tests, and collaborate with experienced engineers."
            ),
            requirements="Enrolled in BS/MS Computer Science. Experience with Python, Git, and REST APIs.",
            benefits="Competitive hourly stipend, engineering mentorship, flexible schedule, and learning budget.",
            telecommuting=0,
            has_company_logo=1,
            has_questions=1,
        )

        # Scam-style ad: Huge income, registration fee, WhatsApp/Gmail, no profile
        scam_result = predictor.predict(
            title="Urgent Data Entry Assistant - Immediate Start!",
            company_profile="",  # Missing company profile
            description=(
                "Work from home and earn $1500 per week guaranteed income! No experience required. "
                "Immediate joining available for limited positions. Send money for refundable "
                "registration fee and background check deposit. Contact recruiter directly on "
                "WhatsApp or personal email at jobrecruiter2026@gmail.com."
            ),
            requirements="No experience required. Laptop and internet connection only.",
            benefits="Get rich quick with daily cash payouts.",
            telecommuting=1,
            has_company_logo=0,
            has_questions=0,
        )

        score_genuine = genuine_result["fraud_probability"]
        score_scam = scam_result["fraud_probability"]

        higher_risk = score_scam > score_genuine
        genuine_label_valid = genuine_result["label"] == label_for_level(genuine_result["risk_level"])
        scam_label_valid = scam_result["label"] == label_for_level(scam_result["risk_level"])

        passed = higher_risk and genuine_label_valid and scam_label_valid

        detail = (
            f"Genuine Ad: {score_genuine*100:.1f}% ({genuine_result['risk_level'].upper()} -> '{genuine_result['label']}'); "
            f"Scam Ad: {score_scam*100:.1f}% ({scam_result['risk_level'].upper()} -> '{scam_result['label']}')"
        )
        print_check("Check 3: Predictor scoring & label alignment", passed, detail)
        return passed
    except Exception as exc:
        print_check("Check 3: Predictor scoring & label alignment", False, f"Exception: {exc}")
        return False


def check_risk_cutoffs_metadata() -> bool:
    """
    Check 4: Asserts that risk cutoffs in models/metadata.json exist and satisfy 0 < low_max < high_min < 1.
    """
    metadata_path = PROJECT_ROOT / "models" / "metadata.json"
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        cutoffs = metadata.get("risk_cutoffs", {})
        low_max = float(cutoffs.get("low_max", -1))
        high_min = float(cutoffs.get("high_min", -1))

        valid = (0.0 < low_max < high_min < 1.0)
        detail = f"low_max={low_max}, high_min={high_min} (Constraint: 0 < low_max < high_min < 1)"
        print_check("Check 4: Calibrated risk cutoffs validity", valid, detail)
        return valid
    except Exception as exc:
        print_check("Check 4: Calibrated risk cutoffs validity", False, f"Failed reading metadata.json: {exc}")
        return False


def check_label_for_level_mapping() -> bool:
    """
    Check 5: Asserts that label_for_level returns exact strings:
    - LOW    -> "Likely genuine"
    - MEDIUM -> "Borderline - review carefully"
    - HIGH   -> "Potentially fraudulent"
    """
    try:
        from predictor import label_for_level

        expected = {
            "low": "Likely genuine",
            "medium": "Borderline - review carefully",
            "high": "Potentially fraudulent",
            "LOW": "Likely genuine",
            "MEDIUM": "Borderline - review carefully",
            "HIGH": "Potentially fraudulent",
        }

        all_match = True
        mismatches: List[str] = []

        for lvl, exp in expected.items():
            act = label_for_level(lvl)
            if act != exp:
                all_match = False
                mismatches.append(f"{lvl}: got '{act}', expected '{exp}'")

        detail = "All level variants mapped to exact required labels." if all_match else "; ".join(mismatches)
        print_check("Check 5: label_for_level mapping accuracy", all_match, detail)
        return all_match
    except Exception as exc:
        print_check("Check 5: label_for_level mapping accuracy", False, f"Exception: {exc}")
        return False


def run_subprocess_test(script_name: str) -> Tuple[bool, str]:
    """
    Runs a standalone script using the current Python executable and captures its output.
    """
    script_path = PROJECT_ROOT / script_name
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        passed = (proc.returncode == 0)
        output_summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        if not passed and proc.stderr:
            output_summary += f" [Stderr: {proc.stderr.strip().splitlines()[-1]}]"
        return passed, output_summary
    except Exception as exc:
        return False, f"Subprocess execution error: {exc}"


def check_subsystem_self_tests() -> bool:
    """
    Check 6: Executes the self-test suites of database.py, rules.py, and reports.py
    as subprocesses using sys.executable and asserts clean exit codes.
    """
    all_passed = True
    subsystems = ["database.py", "rules.py", "reports.py"]

    for script in subsystems:
        passed, summary = run_subprocess_test(script)
        if not passed:
            all_passed = False
        print_check(f"Check 6: Self-test for {script}", passed, summary)

    return all_passed


def check_label_test_script() -> bool:
    """
    Check 7: Executes test_label.py as a subprocess and asserts clean exit code and output.
    """
    passed, summary = run_subprocess_test("test_label.py")
    print_check("Check 7: test_label.py execution", passed, summary)
    return passed


def main() -> int:
    """Main orchestration entry point."""
    print_header("SMOKE TEST SUITE - FAKE JOB POSTING DETECTOR")

    results = [
        check_required_files(),
        check_module_imports(),
        check_predictor_inference(),
        check_risk_cutoffs_metadata(),
        check_label_for_level_mapping(),
        check_subsystem_self_tests(),
        check_label_test_script(),
    ]

    all_passed = all(results)
    print_header("OVERALL SMOKE TEST RESULT")
    if all_passed:
        print("RESULT: ALL 7 SMOKE TESTS PASSED CLEANLY! SYSTEM READY FOR VIVA.")
    else:
        print("RESULT: ONE OR MORE SMOKE TESTS FAILED. PLEASE REVIEW LOGS ABOVE.")
    print("=" * 75)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
