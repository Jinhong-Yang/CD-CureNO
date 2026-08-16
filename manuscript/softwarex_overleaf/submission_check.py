from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "main.tex"
AUTHOR = ROOT / "author_metadata.tex"


def strip_tex(text: str) -> str:
    text = re.sub(r"(?m)(?<!\\)%.*$", " ", text)
    text = re.sub(r"\\begin\{(?:figure|table|lstlisting)\}.*?\\end\{(?:figure|table|lstlisting)\}", " ", text, flags=re.S)
    text = re.sub(r"\\(?:cite|ref|label|url|href|includegraphics|bibliography|bibliographystyle)\*?(?:\[[^\]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}$&_#^~\\]", " ", text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SoftwareX package structure and unresolved human inputs.")
    parser.add_argument("--allow-placeholders", action="store_true", help="Run structural QA without requiring human metadata.")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    required_files = [
        MAIN,
        AUTHOR,
        ROOT / "references.bib",
        ROOT / "elsarticle-num.bst",
        ROOT / "LICENSE.txt",
        ROOT / "CITATION.cff",
        ROOT / "figures" / "p2_case100_temperature.png",
        ROOT / "SOURCE_TRACEABILITY.csv",
        ROOT / "SUBMISSION_CHECKLIST.md",
    ]
    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty required file: {path.relative_to(ROOT)}")

    if errors:
        print("FAIL")
        print("\n".join(f"- {item}" for item in errors))
        return 1

    main_tex = MAIN.read_text(encoding="utf-8")
    author_tex = AUTHOR.read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE.txt").read_text(encoding="utf-8")
    citation_text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        errors.append("LICENSE.txt is not the expected Apache License 2.0 text")
    if "license: Apache-2.0" not in citation_text:
        errors.append("CITATION.cff does not declare Apache-2.0")

    required_sections = [
        "Motivation and significance",
        "Software description",
        "Illustrative examples",
        "Impact",
        "Conclusions",
    ]
    for section in required_sections:
        if f"\\section{{{section}}}" not in main_tex:
            errors.append(f"missing mandatory SoftwareX section: {section}")

    for code in [f"C{i}" for i in range(1, 9)]:
        if not re.search(rf"\b{code}\b", main_tex):
            errors.append(f"missing metadata row {code}")

    figure_count = len(re.findall(r"\\begin\{figure\}", main_tex))
    if figure_count > 6:
        errors.append(f"figure count {figure_count} exceeds SoftwareX maximum of 6")

    stripped = strip_tex(main_tex)
    word_count = len(re.findall(r"\b[A-Za-z][A-Za-z0-9'-]*\b", stripped))
    if word_count > 3000:
        errors.append(f"estimated main-document word count {word_count} exceeds 3000")

    placeholder_count = (main_tex + author_tex).count("SOFTWAREX-REPLACE")
    if placeholder_count and not args.allow_placeholders:
        errors.append(f"{placeholder_count} SOFTWAREX-REPLACE markers remain")

    if ".venv/Scripts" in main_tex:
        errors.append("Windows-specific virtual-environment executable remains in the manuscript")
    if "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process" not in main_tex:
        errors.append("missing generative-AI manuscript-preparation declaration")
    if "/blob/v0.0.2/README.md" not in author_tex:
        errors.append("C7 documentation URL is not pinned to the v0.0.2 README")

    forbidden_claims = ["first-ever", "state-of-the-art", "statistically superior"]
    lower = main_tex.lower()
    for phrase in forbidden_claims:
        if phrase in lower:
            errors.append(f"forbidden or unsupported claim phrase found: {phrase}")

    print("SoftwareX package check")
    print(f"- estimated words: {word_count}")
    print(f"- figure environments: {figure_count}")
    print(f"- unresolved markers: {placeholder_count}")
    for item in warnings:
        print(f"WARNING: {item}")
    if errors:
        print("FAIL")
        for item in errors:
            print(f"- {item}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
