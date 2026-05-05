"""Tests for extractor_parts/text_header.py — _looks_like_author_line guard."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_PROJECT_PARENT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _PACKAGE_DIR]
sys.path.insert(0, str(_PROJECT_PARENT))

from papis_import.extractor_parts.text_header import _looks_like_author_line


class TestLooksLikeAuthorLine(unittest.TestCase):
    # Lines that are clearly NOT author lists.
    REJECT_CASES = [
        # Conjunction / article starts (cases 003, 022)
        "A Hitchhiker's Guide",
        "and Their Applications",
        "of the American Mathematical Society",
        "in Applied Mathematics",
        "by the Authors",
        # Role labels
        "Advisors:",                 # case 016: ends with colon
        "Editors: ",
        "Series Editors",
        "Volume 42",
        # All-caps section labels (no comma)
        "LECTURE NOTES IN MATHEMATICS",
        "APPLIED MATHEMATICAL SCIENCES",
        "GRADUATE TEXTS IN MATHEMATICS",
        # Series page titles
        "Lecture Notes in Mathematics",
        "Graduate Texts in Mathematics",
        "Universitext",
        "Springer Series in Statistics",
        # Empty string
        "",
    ]

    # Lines that look like real author names.
    ACCEPT_CASES = [
        "Charalambos D. Aliprantis",
        "Kim C. Border",
        "Walter Rudin",
        "John B. Conway",
        "M. M. Rao",
        "Erich L. Lehmann, George Casella",   # comma-separated list
        "K. L. Chung",
    ]

    def test_reject(self):
        for line in self.REJECT_CASES:
            with self.subTest(line=line):
                self.assertFalse(
                    _looks_like_author_line(line),
                    f"Expected _looks_like_author_line({line!r}) = False",
                )

    def test_accept(self):
        for line in self.ACCEPT_CASES:
            with self.subTest(line=line):
                self.assertTrue(
                    _looks_like_author_line(line),
                    f"Expected _looks_like_author_line({line!r}) = True",
                )


if __name__ == "__main__":
    unittest.main()
