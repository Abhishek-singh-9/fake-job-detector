"""
Heuristic Rule-Based Checks for Fake Job Posting Detector.

This module provides a deterministic RuleChecker class that scans job postings
for suspicious patterns, known scam indicators, and missing employer data.

Important Note for Viva & Users:
- These heuristic rules are completely SEPARATE from the statistical machine learning model.
- They serve as an immediate advisory checklist for known red flags (e.g. asking for money,
  contacting via WhatsApp, omitting company profiles).
- A triggered rule is an indicator of risk, NEVER definitive proof that an ad is fraudulent.
- A clean rule check does not guarantee legitimacy. The rules and the ML model complement
  one another to provide layered defense.
"""

import html
import re
from typing import Any, Dict, List, Optional, Set, Tuple


class RuleChecker:
    """
    Scans job posting components against heuristic scam rules.

    Each rule produces a standardized warning dictionary with:
    - id: Unique rule identifier
    - severity: 'high', 'medium', or 'low'
    - message: Plain English explanation of the concern
    - matched: Specific phrases or indicators triggered
    """

    # Rule 2: Fee or payment phrases (High severity)
    # Legitimate employers NEVER charge job seekers application fees or training deposits
    FEE_PHRASES: List[str] = [
        "registration fee",
        "security deposit",
        "processing fee",
        "training fee",
        "pay to apply",
        "send money",
        "refundable deposit",
        "application fee",
        "equipment fee",
        "wire transfer",
        "western union",
        "moneygram",
        "cryptocurrency deposit",
    ]

    # Rule 3: Unrealistic earnings or get-rich phrases (High severity)
    # Hyperbolic income promises targeting vulnerable applicants
    EARNINGS_PHRASES: List[str] = [
        "guaranteed income",
        "unlimited earning",
        "work from home and earn",
        "get rich",
        "earn per day",
        "earn per week",
        "daily payout",
        "weekly cash",
        "financial freedom",
        "passive income guarantee",
    ]

    # Rule 4: Urgency wording (Low severity)
    # Artificial time pressure to prevent candidates from verifying details
    URGENCY_PHRASES: List[str] = [
        "urgent",
        "immediate joining",
        "apply now",
        "limited positions",
        "hiring immediately",
        "act now",
        "start today",
        "don't miss out",
    ]

    # Rule 5: Personal email domains or messaging apps (Medium severity)
    # Reputable recruiters use official corporate domains, not anonymous messaging apps
    CONTACT_PHRASES: List[str] = [
        "gmail.com",
        "yahoo.com",
        "hotmail.com",
        "outlook.com",
        "whatsapp",
        "telegram",
        "signal app",
        "viber",
    ]

    # Rule 7: "No experience required" wording (Low severity)
    # Often combined with unusually high pay in entry-level scams
    NO_EXP_PHRASES: List[str] = [
        "no experience required",
        "no experience needed",
        "no experience necessary",
        "zero experience required",
        "no qualifications needed",
    ]

    def __init__(self) -> None:
        """Initializes compiled regex patterns for performance and accuracy."""
        # Regex to detect phrases like 'earn $500 per day', 'earn $1000/week', etc.
        self._earnings_regex = re.compile(
            r"\bearn\s+\$?\d+[\d,]*(?:\s*(?:to|-)\s*\$?\d+[\d,]*)?\s*(?:per|\/)\s*(?:day|week|hr|hour)\b",
            re.IGNORECASE,
        )

    def check(
        self,
        title: str = "",
        company_profile: str = "",
        text: str = "",
        salary_range: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Scans all text fields against the 7 heuristic rules.

        Args:
            title (str): Job title.
            company_profile (str): Description of the hiring company.
            text (str): Main job description / requirements body.
            salary_range (str): Stated compensation range (e.g. 50000-70000).

        Returns:
            List[Dict[str, Any]]: List of warning dicts for triggered rules.
        """
        warnings: List[Dict[str, Any]] = []

        # Combine all textual components into one unified lowercase string for phrase matching
        combined_content: str = f"{title} {company_profile} {text} {salary_range}".lower()

        # ---------------------------------------------------------------------
        # Rule 1: No company profile provided (Medium severity)
        # ---------------------------------------------------------------------
        profile_clean = str(company_profile or "").strip()
        if not profile_clean or profile_clean.lower() == "nan":
            warnings.append({
                "id": "missing_company_profile",
                "severity": "medium",
                "message": "No company profile was provided. Legitimate employers usually describe their business and industry.",
                "matched": ["missing_company_profile"],
            })

        # ---------------------------------------------------------------------
        # Rule 2: Fee or payment wording (High severity)
        # ---------------------------------------------------------------------
        matched_fees: List[str] = [
            phrase for phrase in self.FEE_PHRASES if phrase in combined_content
        ]
        if matched_fees:
            warnings.append({
                "id": "fee_payment_requested",
                "severity": "high",
                "message": "Mention of upfront fees, security deposits, or money transfers. Legitimate employers never charge candidates to apply or work.",
                "matched": matched_fees,
            })

        # ---------------------------------------------------------------------
        # Rule 3: Unrealistic earnings wording (High severity)
        # ---------------------------------------------------------------------
        matched_earnings: Set[str] = {
            phrase for phrase in self.EARNINGS_PHRASES if phrase in combined_content
        }
        # Also check regex pattern (e.g. 'earn $500 per day')
        regex_matches = self._earnings_regex.findall(combined_content)
        for m in regex_matches:
            matched_earnings.add(m)

        if matched_earnings:
            warnings.append({
                "id": "unrealistic_earnings",
                "severity": "high",
                "message": "Promises unrealistic earnings or quick payouts (e.g., daily cash, get-rich promises).",
                "matched": sorted(list(matched_earnings)),
            })

        # ---------------------------------------------------------------------
        # Rule 4: Urgency wording (Low severity)
        # ---------------------------------------------------------------------
        matched_urgency: List[str] = [
            phrase for phrase in self.URGENCY_PHRASES if phrase in combined_content
        ]
        if matched_urgency:
            warnings.append({
                "id": "urgency_pressure",
                "severity": "low",
                "message": "High-urgency language detected. Scammers often create false urgency to pressure candidates into skipping due diligence.",
                "matched": matched_urgency,
            })

        # ---------------------------------------------------------------------
        # Rule 5: Personal or free email / messaging app contact (Medium severity)
        # ---------------------------------------------------------------------
        matched_contacts: List[str] = [
            phrase for phrase in self.CONTACT_PHRASES if phrase in combined_content
        ]
        if matched_contacts:
            warnings.append({
                "id": "personal_contact_methods",
                "severity": "medium",
                "message": "Contact method uses a free public email (Gmail/Yahoo) or messaging app (WhatsApp/Telegram) instead of an official company domain.",
                "matched": matched_contacts,
            })

        # ---------------------------------------------------------------------
        # Rule 6: Very short or vague description (< 30 words) (Medium severity)
        # ---------------------------------------------------------------------
        # Evaluate word count of main description / body
        raw_words = text.strip().split()
        if len(raw_words) < 30:
            warnings.append({
                "id": "short_vague_description",
                "severity": "medium",
                "message": f"Job description is unusually short ({len(raw_words)} words, threshold < 30) and lacks detailed duties.",
                "matched": [f"{len(raw_words)} words"],
            })

        # ---------------------------------------------------------------------
        # Rule 7: 'No experience required' wording (Low severity)
        # ---------------------------------------------------------------------
        matched_no_exp: List[str] = [
            phrase for phrase in self.NO_EXP_PHRASES if phrase in combined_content
        ]
        if matched_no_exp:
            warnings.append({
                "id": "no_experience_required",
                "severity": "low",
                "message": "Listing emphasizes 'no experience required', which is frequently paired with high pay in predatory scam ads.",
                "matched": matched_no_exp,
            })

        return warnings

    def get_keywords(self) -> List[str]:
        """
        Returns a deduplicated list of all phrases monitored across rules.

        Why this matters:
        The Streamlit web UI can use this consolidated keyword list to visually
        highlight warning phrases inside user-submitted job descriptions.

        Returns:
            List[str]: Comprehensive list of all watched scam phrases.
        """
        all_phrases: Set[str] = set()
        all_phrases.update(self.FEE_PHRASES)
        all_phrases.update(self.EARNINGS_PHRASES)
        all_phrases.update(self.URGENCY_PHRASES)
        all_phrases.update(self.CONTACT_PHRASES)
        all_phrases.update(self.NO_EXP_PHRASES)
        return sorted(list(all_phrases))


def highlight_text(
    text: str,
    keywords: Optional[List[str]] = None,
    max_chars: int = 5000,
) -> Tuple[str, int]:
    """
    Sanitizes user text and highlights suspicious scam phrases using HTML <mark> tags.

    Security & Viva Principles:
    1. XSS Defense via html.escape():
       Raw user text is strictly sanitized using `html.escape()` BEFORE any HTML highlight
       tags are injected. Arbitrary HTML or JavaScript tags (e.g. `<script>`, `<b>`, `<img>`)
       are converted into inert text entities (`&lt;script&gt;`, `&lt;b&gt;`). Only the `<mark>`
       tags inserted by this function become executable markup.
    2. Zero Link Following / SSRF Defense:
       Links and URLs in user text are treated strictly as plain text characters. They are
       never fetched, followed, resolved, or wrapped in clickable anchor tags.
    3. Non-Overlapping & Longest-Match First:
       Keywords are sorted in descending order of length (`key=len, reverse=True`).
       Compiling a single regular expression with `re.IGNORECASE` guarantees that:
       - Matches are evaluated case-insensitively.
       - Longer phrases take precedence over shorter substrings (e.g. 'no experience required'
         is matched before 'experience').
       - A single regex scanner advances past matched spans, making nested or broken overlapping
         `<mark>` tags impossible.
    4. Dark Mode High-Contrast Styling:
       Uses an amber background (`#ffd166`) with explicit dark text (`#1a1a1a`), ensuring
       clean readability in both dark and light UI themes.
    5. Paragraph Formatting & Truncation:
       Input length is capped at `max_chars` (default 5,000) with an appended notice,
       and newlines (`\\n`) are converted to `<br>` to maintain layout.

    Args:
        text (str): Raw input job posting text.
        keywords (Optional[List[str]]): List of suspicious phrases to highlight.
            If None, uses `RuleChecker().get_keywords()`.
        max_chars (int): Maximum character length before truncation.

    Returns:
        Tuple[str, int]:
            - highlighted_html (str): Sanitized HTML string with <mark> tags.
            - match_count (int): Total number of highlighted keyword instances found.
    """
    if not text or not str(text).strip():
        return "", 0

    # 1. Truncate if character count exceeds max_chars
    is_truncated = len(text) > max_chars
    content_to_process = text[:max_chars] if is_truncated else text

    # 2. XSS Prevention: escape all HTML characters in user text first
    escaped_text = html.escape(content_to_process)

    # 3. Retrieve and prepare keywords
    if keywords is None:
        keywords = RuleChecker().get_keywords()

    # Escape each keyword and filter empty strings
    clean_keywords = [html.escape(k.strip()) for k in keywords if k and k.strip()]
    if not clean_keywords:
        formatted_html = escaped_text.replace("\n", "<br>")
        if is_truncated:
            formatted_html += (
                '<br><br><span style="color: #888; font-style: italic;">'
                f'[Text truncated after {max_chars:,} characters...]</span>'
            )
        return formatted_html, 0

    # Sort keywords by length in descending order (longest match first)
    sorted_keywords = sorted(list(set(clean_keywords)), key=len, reverse=True)

    # 4. Build non-overlapping regex pattern
    pattern_str = r"(" + "|".join(re.escape(k) for k in sorted_keywords) + r")"
    pattern = re.compile(pattern_str, re.IGNORECASE)

    # 5. Inject <mark> tags with explicit high-contrast styling
    match_count = 0

    def _mark_replacement(match: re.Match) -> str:
        nonlocal match_count
        match_count += 1
        return (
            f'<mark style="background-color: #ffd166; color: #1a1a1a; '
            f'padding: 2px 4px; border-radius: 3px; font-weight: 600;">'
            f'{match.group(0)}</mark>'
        )

    highlighted_content = pattern.sub(_mark_replacement, escaped_text)

    # 6. Preserve paragraph layout by converting newlines to <br>
    highlighted_html = highlighted_content.replace("\n", "<br>")

    if is_truncated:
        highlighted_html += (
            '<br><br><span style="color: #888; font-style: italic;">'
            f'[Text truncated after {max_chars:,} characters...]</span>'
        )

    return highlighted_html, match_count


if __name__ == "__main__":
    # Self-test demonstration when rules.py is run directly
    checker = RuleChecker()
    sample_text = (
        "Urgent! Earn $1000 per week. Pay $50 registration fee via WhatsApp or "
        "send email to test@gmail.com. No experience required."
    )
    results = checker.check(
        title="Work from Home Assistant",
        company_profile="",
        text=sample_text,
        salary_range="",
    )

    print("=" * 65)
    print("RULE CHECKER DEMO")
    print("=" * 65)
    print(f"Total Monitored Scam Keywords/Phrases: {len(checker.get_keywords())}")
    print(f"Warnings Triggered on Test Ad:         {len(results)}")
    for w in results:
        print(f"  - [{w['severity'].upper()}] {w['id']}: {w['message']}")
        print(f"    Matched: {w['matched']}")
    print("=" * 65)

    print("\n" + "=" * 65)
    print("RUNNING HIGHLIGHT_TEXT SELF-TEST SUITE")
    print("=" * 65)

    # (a) A phrase is wrapped in <mark>
    test_text_a = "Please submit the registration fee to complete your application."
    html_a, cnt_a = highlight_text(test_text_a)
    test_a_pass = (
        cnt_a >= 1
        and "<mark" in html_a
        and "registration fee</mark>" in html_a
    )
    print(f"Test (a) Phrase wrapped in <mark>:         {'[PASS]' if test_a_pass else '[FAIL]'}")
    assert test_a_pass, "Test (a) failed: phrase not wrapped in <mark>"

    # (b) Matching is case-insensitive
    test_text_b = "URGENT hiring! IMMEDIATE JOINING available for all positions."
    html_b, cnt_b = highlight_text(test_text_b)
    test_b_pass = (
        cnt_b >= 2
        and "URGENT</mark>" in html_b
        and "IMMEDIATE JOINING</mark>" in html_b
    )
    print(f"Test (b) Case-insensitive matching:        {'[PASS]' if test_b_pass else '[FAIL]'}")
    assert test_b_pass, "Test (b) failed: case-insensitive matching not working"

    # (c) "<b>" in the input is escaped (XSS Defense)
    test_text_c = "Job posting with <b>bold formatting</b> and <script>alert('xss')</script>."
    html_c, cnt_c = highlight_text(test_text_c)
    test_c_pass = (
        "<b>" not in html_c
        and "<script>" not in html_c
        and "&lt;b&gt;" in html_c
        and "&lt;script&gt;" in html_c
    )
    print(f"Test (c) Raw <b> and <script> escaped:     {'[PASS]' if test_c_pass else '[FAIL]'}")
    assert test_c_pass, "Test (c) failed: HTML tags not escaped"

    # (d) Overlapping phrases do not break the HTML
    custom_keywords = ["registration fee", "fee", "earn cash", "cash"]
    test_text_d = "Please pay registration fee to earn cash immediately."
    html_d, cnt_d = highlight_text(test_text_d, keywords=custom_keywords)
    mark_opens = html_d.count("<mark")
    mark_closes = html_d.count("</mark>")
    has_nested_mark = bool(re.search(r"<mark[^>]*>[^<]*<mark", html_d))
    test_d_pass = (
        mark_opens == 2
        and mark_closes == 2
        and not has_nested_mark
        and "registration fee</mark>" in html_d
        and "earn cash</mark>" in html_d
    )
    print(f"Test (d) Overlapping phrases HTML integrity: {'[PASS]' if test_d_pass else '[FAIL]'}")
    assert test_d_pass, "Test (d) failed: overlapping phrases broke HTML"

    print("=" * 65)
    print("ALL HIGHLIGHT_TEXT SELF-TESTS PASSED CLEANLY!")
    print("=" * 65)

