from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from email.message import Message
from pathlib import Path
from unittest import mock

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_PROJECT_PARENT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _PACKAGE_DIR]
sys.path.insert(0, str(_PROJECT_PARENT))

import urllib.error

from papis_import.http.cache import Cache
from papis_import.http_client import HttpClient
from papis_import.output.writers import LiveStatusReporter


class _FakeResponse:
    status = 200

    def __init__(self, body: bytes = b'{"ok": true}') -> None:
        self.headers = Message()
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _http_error(status: int, body: bytes, headers: Message | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.example.test/v1/chat/completions?api_key=secret",
        status,
        "rate limited",
        headers or Message(),
        io.BytesIO(body),
    )


class LiveStatusTests(unittest.TestCase):
    def test_live_status_writer_writes_valid_atomic_json_and_omits_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "current_status.json"
            reporter = LiveStatusReporter(path)

            reporter.file_started(3, 10, Path("/tmp/example.pdf"))
            reporter.update(
                status="sleeping",
                event="rate_limit_sleep",
                wait_s=12,
                request_body={"prompt": "must not be written"},
                api_key="must-not-be-written",
            )

            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "sleeping")
            self.assertEqual(data["idx"], 3)
            self.assertEqual(data["file_name"], "example.pdf")
            self.assertNotIn("request_body", data)
            self.assertNotIn("api_key", data)
            self.assertFalse(list(Path(td).glob("*.tmp")))

    def test_http_429_retry_reports_before_sleep(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            status_path = Path(td) / "status.json"
            reporter = LiveStatusReporter(status_path)
            reporter.file_started(1, 1, Path("/tmp/paper.pdf"))
            http = HttpClient(Cache(str(Path(td) / "cache")), live_reporter=reporter)
            headers = Message()
            headers["x-ratelimit-remaining-tokens"] = "0"
            headers["x-ratelimit-reset-tokens"] = "2s"
            body = b'{"error":{"message":"Rate limit reached. Please try again in 2s","type":"rate_limit_error"}}'

            calls = [_http_error(429, body, headers), _FakeResponse()]

            def fake_urlopen(*args: object, **kwargs: object) -> object:
                item = calls.pop(0)
                if isinstance(item, urllib.error.HTTPError):
                    raise item
                return item

            stderr = io.StringIO()
            sleep_status: dict[str, object] = {}

            def fake_sleep(seconds: float) -> None:
                if seconds == 2:
                    sleep_status.update(json.loads(status_path.read_text(encoding="utf-8")))

            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen), \
                    mock.patch("time.sleep", side_effect=fake_sleep) as sleep_mock, \
                    redirect_stderr(stderr):
                result = http.post_json(
                    "https://api.example.test/v1/chat/completions?api_key=secret",
                    {"messages": [{"content": "secret prompt"}]},
                    "llm",
                    "retry-test",
                    bucket="llm",
                    max_retries=2,
                    min_remaining_tokens=1,
                )

            self.assertEqual(result, {"ok": True})
            self.assertIn("[retry] llm POST HTTP 429", stderr.getvalue())
            self.assertIn("url=https://api.example.test/v1/chat/completions", stderr.getvalue())
            self.assertNotIn("api_key=secret", stderr.getvalue())
            sleep_mock.assert_any_call(2)

            self.assertEqual(sleep_status["event"], "rate_limit_sleep")
            self.assertEqual(sleep_status["status"], "sleeping")
            self.assertEqual(sleep_status["sleep_reason"], "rate_limit")
            self.assertEqual(sleep_status["attempt"], 1)
            self.assertEqual(sleep_status["http_status"], 429)
            self.assertIn("x-ratelimit-remaining-tokens", sleep_status["rate_limit"])

    def test_proactive_token_pacing_reports_before_sleep(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            status_path = Path(td) / "status.json"
            reporter = LiveStatusReporter(status_path)
            reporter.file_started(1, 1, Path("/tmp/paper.pdf"))
            http = HttpClient(Cache(str(Path(td) / "cache")), live_reporter=reporter)
            http._token_budget["vision_llm"] = (5, 1000.0)

            stderr = io.StringIO()
            with mock.patch("time.monotonic", return_value=990.0), \
                    mock.patch("time.sleep") as sleep_mock, \
                    redirect_stderr(stderr):
                slept = http._wait_for_token_budget("vision_llm", 10_000)

            self.assertEqual(slept, 11.0)
            sleep_mock.assert_called_once_with(11.0)
            self.assertIn("[pacing] vision_llm: 5 tokens remaining", stderr.getvalue())

            data = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(data["event"], "pacing_sleep")
            self.assertEqual(data["status"], "sleeping")
            self.assertEqual(data["remaining_tokens"], 5)
            self.assertEqual(data["needed_tokens"], 10000)


if __name__ == "__main__":
    unittest.main()
