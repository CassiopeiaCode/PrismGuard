from pathlib import Path
import importlib.util
import sys
import unittest


def load_module():
    script_path = Path(__file__).resolve().parents[1] / "bench_inline_moderation.py"
    spec = importlib.util.spec_from_file_location("bench_inline_moderation", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BenchInlineModerationTests(unittest.TestCase):
    def test_build_inline_proxy_path_uses_inline_source_and_profile(self):
        mod = load_module()
        upstream = "http://127.0.0.1:19001/v1/chat/completions"
        path = mod.build_inline_proxy_path(
            upstream_url=upstream,
            profile="4claudecode",
            stream=False,
        )

        self.assertTrue(path.startswith("/"))
        self.assertIn("$http://127.0.0.1:19001/v1/chat/completions", path)
        self.assertNotIn("\"source\":\"inline\"", path)
        self.assertIn("4claudecode", path)
        self.assertIn("smart_moderation", path)

    def test_discover_base_url_prefers_proc_env_then_healthz(self):
        mod = load_module()

        mod.find_prismguard_pids = lambda: [123]
        mod.read_proc_environ = lambda pid: {"HOST": "127.0.0.1", "PORT": "8123"}
        mod.load_dotenv_values = lambda path: {"HOST": "127.0.0.1", "PORT": "9999"}
        mod.probe_healthz = lambda url, timeout_secs: (url == "http://127.0.0.1:8123", 0.012)

        result = mod.discover_base_url(repo_root=Path("/repo"), timeout_secs=1.0)

        self.assertEqual(result.base_url, "http://127.0.0.1:8123")
        self.assertTrue(any(item.startswith("proc:123") for item in result.trace))

    def test_mock_upstream_serves_json_and_sse(self):
        mod = load_module()

        with mod.MockUpstreamServer(
            json_delay_ms=5,
            stream_first_token_delay_ms=5,
            stream_tail_delay_ms=5,
        ) as server:
            json_result = mod.fetch_json(f"{server.base_url}/v1/chat/completions", {"stream": False})
            stream_result = mod.fetch_sse(
                f"{server.base_url}/v1/chat/completions",
                {"stream": True},
                timeout_secs=2.0,
            )

        self.assertEqual(json_result["choices"][0]["message"]["content"], "mock-ok")
        self.assertTrue(stream_result.done_seen)
        self.assertGreaterEqual(stream_result.first_event_latency_ms, 0.0)

    def test_run_non_stream_benchmark_reports_proxy_and_upstream_latency(self):
        mod = load_module()

        mod.load_request_payload = lambda request_file: {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hello"}],
        }
        mod.fetch_proxy_json = lambda *args, **kwargs: mod.ProxyJsonResult(
            status_code=200,
            total_latency_ms=42.0,
            body={"choices": [{"message": {"content": "ok"}}]},
            error_body=None,
        )

        result = mod.run_non_stream_benchmark(
            base_url="http://127.0.0.1:8000",
            upstream_url="http://127.0.0.1:19001/v1/chat/completions",
            profile="4claudecode",
            request_file=None,
            timeout_secs=3.0,
            upstream_delay_ms=11,
        )

        self.assertEqual(result.mode, "non-stream")
        self.assertEqual(result.total_latency_ms, 42.0)
        self.assertEqual(result.proxy_overhead_ms, 31.0)

    def test_run_stream_benchmark_reports_first_event_and_done(self):
        mod = load_module()

        mod.load_request_payload = lambda request_file: {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hello"}],
        }
        mod.fetch_proxy_sse = lambda *args, **kwargs: mod.StreamCapture(
            status_code=200,
            first_event_latency_ms=18.0,
            total_latency_ms=55.0,
            done_seen=True,
            lines=["event: response.created", "data: [DONE]"],
        )

        result = mod.run_stream_benchmark(
            base_url="http://127.0.0.1:8000",
            upstream_url="http://127.0.0.1:19001/v1/chat/completions",
            profile="4claudecode",
            request_file=None,
            timeout_secs=3.0,
            first_token_delay_ms=7,
            tail_delay_ms=13,
        )

        self.assertEqual(result.mode, "stream")
        self.assertEqual(result.total_latency_ms, 55.0)
        self.assertEqual(result.first_event_latency_ms, 18.0)
        self.assertTrue(result.done_seen)

    def test_prepare_request_payload_randomizes_only_user_text(self):
        mod = load_module()

        payload = {
            "model": "gpt-4.1-mini",
            "messages": [
                {"role": "system", "content": "stay constant"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "previous answer"},
            ],
            "metadata": {"keep": "same"},
        }

        prepared, nonce = mod.prepare_request_payload(payload, randomize_user_text=True, nonce="fixed-nonce")

        self.assertEqual(nonce, "fixed-nonce")
        self.assertEqual(prepared["messages"][0]["content"], "stay constant")
        self.assertEqual(prepared["messages"][2]["content"], "previous answer")
        self.assertEqual(prepared["metadata"], {"keep": "same"})
        self.assertIn("benchmark_nonce=fixed-nonce", prepared["messages"][1]["content"])
        self.assertTrue(prepared["messages"][1]["content"].startswith("hello"))

    def test_prepare_request_payload_can_disable_user_text_randomization(self):
        mod = load_module()

        payload = {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hello"}],
        }

        prepared, nonce = mod.prepare_request_payload(payload, randomize_user_text=False, nonce="fixed-nonce")

        self.assertIsNone(nonce)
        self.assertEqual(prepared, payload)


if __name__ == "__main__":
    unittest.main()
