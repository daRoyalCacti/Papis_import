"""Tests for core/title_quality.py — series-page detection and pdfinfo filters."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_PROJECT_PARENT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _PACKAGE_DIR]
sys.path.insert(0, str(_PROJECT_PARENT))

from papis_import.core.title_quality import (
    is_garbage_pdfinfo_author,
    is_garbage_pdfinfo_title,
    is_series_page_title,
)


class TestIsSeriesPageTitle(unittest.TestCase):
    TRUE_CASES = [
        # Springer series patterns
        "Lecture Notes in Mathematics",
        "Lecture Notes in Physics",
        "Lecture Notes in Computer Science",
        "Lecture Notes in Statistics",
        "lecture notes in artificial intelligence",  # lowercase
        "Springer Series in Statistics",
        "Springer Series in Computational Mathematics",
        "Springer Monographs in Mathematics",
        "Springer Tracts in Modern Physics",
        # Other publisher series
        "Studies in Logic and the Foundations of Mathematics",
        "Progress in Nonlinear Differential Equations and Their Applications",
        "Progress in Mathematics",
        "Progress in Theoretical Computer Science",
        "Modeling and Simulation in Science, Engineering and Technology",
        "Modeling and Simulation in Engineering",
        "Wiley Classics Library",
        "Graduate Texts in Mathematics",
        "Graduate Texts in Physics",
        "Graduate Texts in Statistics",
        "Universitext",
        "Subseries of Lecture Notes in Mathematics",
        "Applied Mathematical Sciences",
        "Grundlehren der mathematischen Wissenschaften",
        "Ergebnisse der Mathematik und ihrer Grenzgebiete",
        "North-Holland Mathematical Library",
        "North Holland Mathematical Library",
        "Cambridge Studies in Advanced Mathematics",
        "London Mathematical Society Lecture Note Series",
        "Oxford Lecture Series in Mathematics and its Applications",
        "De Gruyter Studies in Mathematics",
        "Texts in Applied Mathematics",
        "Pure and Applied Mathematics",
    ]

    FALSE_CASES = [
        # Real book titles
        "Infinite Dimensional Analysis: A Hitchhiker's Guide",
        "An Introduction to Γ-Convergence",
        "Introduction to Nonlinear Differential Equations",
        "The Theory of Groups",
        "Functional Analysis",
        "Real and Complex Analysis",
        "Probability and Measure",
        "Markov Chains",
        # Short strings
        "Mathematics",
        "Lecture",
        "",
        # Paper titles
        "On the convergence of stochastic gradient descent",
        "Attention Is All You Need",
    ]

    def test_true_cases(self):
        for title in self.TRUE_CASES:
            with self.subTest(title=title):
                self.assertTrue(
                    is_series_page_title(title),
                    f"Expected is_series_page_title({title!r}) to be True",
                )

    def test_false_cases(self):
        for title in self.FALSE_CASES:
            with self.subTest(title=title):
                self.assertFalse(
                    is_series_page_title(title),
                    f"Expected is_series_page_title({title!r}) to be False",
                )


class TestIsGarbagePdfinfoTitle(unittest.TestCase):
    TRUE_CASES = [
        "77111_4_En_Print.indd",
        "chapter1.tex",
        "document.pdf",
        "ISO 15930-1: Graphic technology — Prepress digital data exchange",
        "",
    ]

    FALSE_CASES = [
        "Introduction to Banach Spaces",
        "Stochastic Processes",
    ]

    def test_true_cases(self):
        for title in self.TRUE_CASES:
            with self.subTest(title=title):
                self.assertTrue(is_garbage_pdfinfo_title(title))

    def test_false_cases(self):
        for title in self.FALSE_CASES:
            with self.subTest(title=title):
                self.assertFalse(is_garbage_pdfinfo_title(title))


class TestIsGarbagePdfinfoAuthor(unittest.TestCase):
    TRUE_CASES = [
        "ufonter",
        "pdflatex",
        "operator",
        "texlive",
        "ISO 15930-1 WG",
        "cover_page.indd",
        "",
        "   ",
    ]

    FALSE_CASES = [
        "Walter Rudin",
        "Charalambos D. Aliprantis",
        "Kim C. Border",
        "John B. Conway",
    ]

    def test_true_cases(self):
        for author in self.TRUE_CASES:
            with self.subTest(author=author):
                self.assertTrue(is_garbage_pdfinfo_author(author))

    def test_false_cases(self):
        for author in self.FALSE_CASES:
            with self.subTest(author=author):
                self.assertFalse(is_garbage_pdfinfo_author(author))


if __name__ == "__main__":
    unittest.main()
