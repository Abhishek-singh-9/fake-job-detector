"""
Verification script for 3-tier risk classification labels.

Ensures:
1. label_for_level("low") == "Likely genuine"
2. label_for_level("medium") == "Borderline - review carefully"
3. label_for_level("high") == "Potentially fraudulent"
4. Predictor().predict(...) with HTML and registration fee correctly predicts
   medium risk and yields "Borderline - review carefully".
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictor import Predictor, label_for_level


def run_tests() -> bool:
    all_passed = True
    print("=" * 70)
    print("RUNNING CLASSIFICATION LABEL TESTS (test_label.py)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Unit Tests for label_for_level()
    # -------------------------------------------------------------------------
    test_cases = [
        ("low", "Likely genuine"),
        ("medium", "Borderline - review carefully"),
        ("high", "Potentially fraudulent"),
        ("LOW", "Likely genuine"),
        ("Medium", "Borderline - review carefully"),
        ("HIGH", "Potentially fraudulent"),
    ]

    print("\n--- 1. Testing label_for_level() mappings ---")
    for level_input, expected in test_cases:
        actual = label_for_level(level_input)
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"[{status}] label_for_level('{level_input}') -> '{actual}' (expected: '{expected}')")

    # -------------------------------------------------------------------------
    # Integration Test for Predictor().predict(...)
    # -------------------------------------------------------------------------
    print("\n--- 2. Testing Predictor().predict(...) with borderline input ---")
    predictor = Predictor()

    test_description = "<b>bold</b> & <script>x</script> Pay a registration fee today."
    test_company_profile = ""

    result = predictor.predict(
        description=test_description,
        company_profile=test_company_profile,
    )

    risk_score_pct = round(result["fraud_probability"] * 100, 2)
    risk_level = result["risk_level"]
    label = result["label"]

    print(f"Risk Score:        {risk_score_pct}% ({result['fraud_probability']})")
    print(f"Risk Level:        {risk_level}")
    print(f"Classification:    {label}")

    # Check 1: Risk level should be medium for this text
    if risk_level == "medium":
        print(f"[PASS] Risk level is correctly 'medium' for 40%-49% score.")
    else:
        print(f"[FAIL] Unexpected risk level: '{risk_level}' (expected 'medium').")
        all_passed = False

    # Check 2: Label must be "Borderline - review carefully"
    expected_medium_label = "Borderline - review carefully"
    if label == expected_medium_label:
        print(f"[PASS] Classification label for medium risk is '{expected_medium_label}'.")
    else:
        print(f"[FAIL] Classification label is '{label}', expected '{expected_medium_label}'.")
        all_passed = False

    print("=" * 70)
    if all_passed:
        print("OVERALL RESULT: ALL CHECKS PASSED")
    else:
        print("OVERALL RESULT: SOME CHECKS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
