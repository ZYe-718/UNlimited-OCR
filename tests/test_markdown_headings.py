"""
Verification test for Markdown headings structure and hierarchy across workspace .md files.
"""

import os
import re
import sys


def check_markdown_file(filepath: str) -> list[str]:
    errors = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    last_level = 0
    in_code_block = False

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Track fenced code blocks
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        # Check for raw HTML heading tags (<h1..6>)
        if re.search(r"<h[1-6][\s>]", line, re.IGNORECASE):
            errors.append(f"{filepath}:{idx} - Raw HTML heading tag found: {line.strip()}")

        # Check Markdown headings hierarchy
        heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading_match:
            level = len(heading_match.group(1))
            if last_level > 0 and level > last_level + 1:
                errors.append(
                    f"{filepath}:{idx} - Heading level jump detected (H{last_level} -> H{level}): '{line.strip()}'"
                )
            last_level = level

    return errors


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_errors = []

    for item in os.listdir(root_dir):
        if item.endswith(".md"):
            path = os.path.join(root_dir, item)
            errs = check_markdown_file(path)
            all_errors.extend(errs)

    if all_errors:
        print("Markdown Heading Verification Failed:")
        for err in all_errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("All Markdown heading verification checks passed successfully!")


if __name__ == "__main__":
    main()
