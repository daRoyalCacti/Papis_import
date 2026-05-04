from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_PROJECT_PARENT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _PACKAGE_DIR]
sys.path.insert(0, str(_PROJECT_PARENT))

from papis_import.http.cache import Cache
from papis_import.http_client import HttpClient
from papis_import.llm_config import local_llm_response_is_cacheable, text_llm_config, vision_llm_config
from papis_import.ollama_service import ensure_model, local_ollama_session


class _FakeResponse:
    status = 200
    headers: dict[str, str] = {}

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class _FakeProcess:
    pid = 12345

    def __init__(self) -> None:
        self.waited = False

    def poll(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> None:
        self.waited = True
        return None


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "local_llm": True,
        "local_llm_model": "qwen3-vl:8b",
        "local_llm_num_predict": 2048,
        "ollama_model": "",
        "ollama_host": "http://127.0.0.1:11434",
        "ollama_start_timeout": 60,
        "ollama_request_timeout": 300.0,
        "ollama_pull_missing": False,
        "verbose": False,
        "llm_endpoint": "https://api.example.test/v1",
        "llm_model": "remote-text",
        "llm_api_key": "secret",
        "vision_llm_endpoint": "https://vision.example.test/v1",
        "vision_llm_model": "remote-vision",
        "vision_llm_api_key": "vision-secret",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class LocalLlmConfigTests(unittest.TestCase):
    def test_local_llm_overrides_text_and_vision_endpoint_and_disables_tokens(self) -> None:
        args = _args()

        text = text_llm_config(args)
        vision = vision_llm_config(args)

        self.assertEqual(text.endpoint, "http://127.0.0.1:11434")
        self.assertEqual(vision.endpoint, "http://127.0.0.1:11434")
        self.assertEqual(text.model, "qwen3-vl:8b")
        self.assertEqual(vision.model, "qwen3-vl:8b")
        self.assertFalse(text.track_tokens)
        self.assertFalse(vision.track_tokens)
        self.assertEqual(text.timeout_s, 300.0)
        self.assertEqual(text.max_tokens, 2048)
        self.assertIs(text.think, False)

    def test_remote_config_is_unchanged_without_local_flag(self) -> None:
        args = _args(local_llm=False)

        text = text_llm_config(args)
        vision = vision_llm_config(args)

        self.assertEqual(text.endpoint, "https://api.example.test/v1")
        self.assertEqual(text.model, "remote-text")
        self.assertTrue(text.track_tokens)
        self.assertEqual(vision.endpoint, "https://vision.example.test/v1")
        self.assertEqual(vision.model, "remote-vision")


class OllamaServiceTests(unittest.TestCase):
    def test_missing_model_errors_by_default_with_pull_hint(self) -> None:
        body = json.dumps({"models": [{"name": "other:latest"}]}).encode()
        with mock.patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
            with self.assertRaisesRegex(RuntimeError, "ollama pull qwen3-vl:8b"):
                ensure_model("http://127.0.0.1:11434", "qwen3-vl:8b", pull_missing=False)

    def test_missing_model_pulls_when_opted_in(self) -> None:
        body = json.dumps({"models": []}).encode()
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch("urllib.request.urlopen", return_value=_FakeResponse(body)), \
                mock.patch("subprocess.run", return_value=completed) as run:
            ensure_model("http://127.0.0.1:11434", "qwen3-vl:8b", pull_missing=True)

        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["ollama", "pull", "qwen3-vl:8b"])

    def test_session_reuses_running_service_without_stopping_it(self) -> None:
        args = _args()
        body = json.dumps({"models": [{"name": "qwen3-vl:8b"}]}).encode()
        with mock.patch("urllib.request.urlopen", return_value=_FakeResponse(body)), \
                mock.patch("subprocess.Popen") as popen, \
                mock.patch("papis_import.ollama_service.stop_ollama") as stop:
            with local_ollama_session(args) as session:
                self.assertFalse(session.started_here)
                self.assertEqual(session.model, "qwen3-vl:8b")

        popen.assert_not_called()
        stop.assert_not_called()

    def test_session_starts_and_stops_only_started_process(self) -> None:
        args = _args()
        process = _FakeProcess()
        states = [False, True]
        body = json.dumps({"models": [{"name": "qwen3-vl:8b"}]}).encode()

        def fake_alive(*args: object, **kwargs: object) -> bool:
            return states.pop(0) if states else True

        with mock.patch("papis_import.ollama_service.ollama_is_alive", side_effect=fake_alive), \
                mock.patch("papis_import.ollama_service.command_exists", return_value=True), \
                mock.patch("subprocess.Popen", return_value=process), \
                mock.patch("urllib.request.urlopen", return_value=_FakeResponse(body)), \
                mock.patch("os.killpg") as killpg:
            with local_ollama_session(args) as session:
                self.assertTrue(session.started_here)
                self.assertIs(session.process, process)

        killpg.assert_called_once()


class HttpClientLocalRequestTests(unittest.TestCase):
    def test_post_json_can_disable_token_tracking_and_set_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            http = HttpClient(Cache(str(Path(td) / "cache")))
            seen_timeout: list[float] = []

            def fake_urlopen(req: object, timeout: float) -> _FakeResponse:
                seen_timeout.append(timeout)
                return _FakeResponse(b'{"ok": true}')

            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen), \
                    mock.patch.object(http, "_wait_for_token_budget") as wait:
                result = http.post_json(
                    "http://127.0.0.1:11434/v1/chat/completions",
                    {"model": "qwen3-vl:8b"},
                    "llm",
                    "local-timeout-test",
                    timeout_s=321.0,
                    track_tokens=False,
                )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(seen_timeout, [321.0])
        wait.assert_not_called()

    def test_length_limited_llm_responses_are_not_cacheable(self) -> None:
        self.assertFalse(local_llm_response_is_cacheable({"choices": [{"finish_reason": "length"}]}))
        self.assertFalse(local_llm_response_is_cacheable({"done_reason": "length"}))


if __name__ == "__main__":
    unittest.main()
