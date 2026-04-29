from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from papis_import.core.identifiers import extract_identifiers
from papis_import.core.text import clean_text, first_year
from papis_import.core.title_quality import is_suspicious_title
from papis_import.http_client import HttpClient
from papis_import.models import Candidate


class GrobidExtractor:
    def __init__(self, args: argparse.Namespace, http: HttpClient) -> None:
        self.args = args
        self.http = http

    def candidate(self, path: Path) -> list[Candidate]:
        if not self.args.grobid_url:
            return []
        base = self.args.grobid_url.rstrip("/")
        cache_key = f"{path.resolve()}::{path.stat().st_mtime_ns}"
        tei = self.http.post_multipart(
            f"{base}/api/processHeaderDocument",
            fields={"consolidateHeader": "1"},
            file_field="input",
            file_path=path,
            namespace="grobid",
            cache_key=cache_key,
        )
        if not tei.strip():
            return []
        try:
            root = ET.fromstring(tei)
        except Exception:
            return []
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        title = clean_text(" ".join((root.findtext(".//tei:titleStmt/tei:title", default="", namespaces=ns) or "").split()))
        authors: list[str] = []
        for auth in root.findall(".//tei:sourceDesc//tei:author", ns):
            forename = clean_text(auth.findtext(".//tei:forename", default="", namespaces=ns) or "")
            surname = clean_text(auth.findtext(".//tei:surname", default="", namespaces=ns) or "")
            full = clean_text(" ".join(x for x in (forename, surname) if x))
            if full:
                authors.append(full)
        blob = clean_text(" ".join(root.itertext()))
        dois, isbns, arxivs, _ = extract_identifiers(blob)
        if title and is_suspicious_title(title):
            title = ""
        if not (title or authors or dois or isbns or arxivs):
            return []
        return [
            Candidate(
                title=title,
                authors=authors,
                year=first_year(blob),
                doi=dois[0] if dois else "",
                isbn=isbns[0] if isbns else "",
                arxiv=arxivs[0] if arxivs else "",
                source="grobid",
                priority=12,
            )
        ]
