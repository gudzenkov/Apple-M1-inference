from __future__ import annotations

import unittest
from argparse import Namespace

from src.bench.composition import compose_benchmark_spec
from src.bench.process import _render_start_cmd
from src.bench.metrics.openai import _perf_and_usage_from_llama_timings
from src.bench.runner.execution import _tokenizer_model_id_for_runtime_model
from src.bench.runner.execution import _build_prime_case
from src.bench.runner.stats import row_prefill_sec
from src.bench.runner.summary import runtime_summary_rows


class BenchStatsTest(unittest.TestCase):
    def test_build_prime_case_prefill_keeps_real_case_payload(self) -> None:
        case = {
            "case_name": "long-64k-1",
            "prompt": "REAL FULL PROMPT",
            "prompt_suffix": "Question: real",
            "max_tokens": 128,
            "needle_key": "needle",
            "needle_value": "value",
            "needle_position": 123,
        }

        prime_case = _build_prime_case(case, cache_mode="prefill")

        self.assertEqual(prime_case["case_name"], "long-64k-0")
        self.assertEqual(prime_case["phase"], "cache-prime")
        self.assertEqual(prime_case["prompt"], "REAL FULL PROMPT")
        self.assertEqual(prime_case["prompt_suffix"], "Question: real")
        self.assertEqual(prime_case["max_tokens"], 128)
        self.assertEqual(prime_case["needle_key"], "needle")

    def test_build_prime_case_request_uses_synthetic_cache_prime_prompt(self) -> None:
        case = {
            "case_name": "long-64k-1",
            "prompt": "REAL FULL PROMPT",
            "prompt_prefix": "REAL PREFIX\n",
            "prompt_suffix": "Question: real",
            "max_tokens": 128,
            "needle_key": "needle",
            "needle_value": "value",
            "needle_position": 123,
        }

        prime_case = _build_prime_case(case, cache_mode="request")

        self.assertEqual(prime_case["case_name"], "long-64k-0")
        self.assertEqual(prime_case["phase"], "cache-prime")
        self.assertEqual(prime_case["max_tokens"], 8)
        self.assertEqual(
            prime_case["prompt_suffix"],
            "Question: Return exactly CACHE-PRIME-ONLY\nAnswer format: CACHE-PRIME-ONLY",
        )
        self.assertNotIn("needle_key", prime_case)
        self.assertNotIn("needle_value", prime_case)
        self.assertNotIn("needle_position", prime_case)

    def test_row_prefill_sec_prefers_explicit_cache_prefill_timing(self) -> None:
        row = {
            "phase": "cache-prime",
            "timing": {
                "client": {"total_time_sec": 4.6975},
                "cache": {"prefill_sec": 361.2345},
            },
        }

        self.assertEqual(row_prefill_sec(row), 361.2345)

    def test_runtime_summary_uses_explicit_prefill_for_prime_rows(self) -> None:
        results = [
            {
                "success": True,
                "runtime": "mlx",
                "phase": "cache-prime",
                "benchmark_included": False,
                "timing": {
                    "client": {"total_time_sec": 4.6975},
                    "cache": {"prefill_sec": 361.2345},
                    "server": {"ttft_sec": 0.7},
                },
                "throughput": {"prompt_tps": 93.7391, "generation_tps": 32.3839},
                "usage": {"normalized": {"prompt_tokens": 46}},
                "memory": {"peak_gb": 12.9432},
                "retrieval": {"score_float": 1.0, "exact": True},
            },
            {
                "success": True,
                "runtime": "mlx",
                "phase": "benchmark",
                "benchmark_included": True,
                "timing": {
                    "client": {"total_time_sec": 5.0179},
                    "server": {"ttft_sec": 0.696},
                },
                "throughput": {
                    "tokens_per_second": 25.5088,
                    "prompt_tps": 91.9648,
                    "generation_tps": 30.0667,
                },
                "usage": {"normalized": {"prompt_tokens": 46}},
                "memory": {"peak_gb": 12.9432},
                "retrieval": {"score_float": 1.0, "exact": True},
            },
        ]

        rows = runtime_summary_rows(results, ["mlx"])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["avg_prefill_sec"], 361.235)

    def test_render_start_cmd_substitutes_model_host_and_port(self) -> None:
        rendered = _render_start_cmd(
            ["llama-server", "--hf-repo", "{model}", "--host", "{host}", "--port", "{port}"],
            model="unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
            port=8090,
        )

        self.assertEqual(
            rendered,
            [
                "llama-server",
                "--hf-repo",
                "unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
                "--host",
                "127.0.0.1",
                "--port",
                "8090",
            ],
        )

    def test_compose_benchmark_spec_for_llama_cpp_uses_request_cache_defaults(self) -> None:
        args = Namespace(
            reasoning_mode="auto",
            cache_mode="auto",
            transport="auto",
            stream="auto",
            request_timeout=None,
            server_start_timeout=None,
            request_options=None,
        )

        spec = compose_benchmark_spec(
            runtime="llama.cpp",
            model="llama-cpp-qwen-9b",
            args=args,
        )

        self.assertEqual(spec.runtime, "llama.cpp")
        self.assertEqual(spec.model, "unsloth/Qwen3.5-9B-GGUF:Q4_K_M")
        self.assertEqual(spec.cache_mode, "request")
        self.assertEqual(spec.transport_mode, "openai-compat")
        self.assertTrue(spec.stream_enabled)
        self.assertEqual(spec.port, 8090)
        self.assertTrue(spec.managed_server)

    def test_perf_and_usage_from_llama_timings_uses_native_server_fields(self) -> None:
        perf, usage = _perf_and_usage_from_llama_timings(
            {
                "prompt_n": 509,
                "prompt_ms": 1602.852,
                "prompt_per_second": 317.55895116954025,
                "predicted_n": 64,
                "predicted_ms": 2181.829,
                "predicted_per_second": 29.333187889610045,
            }
        )

        self.assertEqual(
            perf,
            {
                "ttft_sec": 1.6029,
                "total_time_sec": 3.7847,
                "prompt_tps": 317.559,
                "generation_tps": 29.3332,
            },
        )
        self.assertEqual(
            usage,
            {
                "prompt_tokens": 509,
                "completion_tokens": 64,
                "total_tokens": 573,
            },
        )

    def test_llama_cpp_model_uses_explicit_tokenizer_source(self) -> None:
        self.assertEqual(
            _tokenizer_model_id_for_runtime_model(
                runtime="llama.cpp",
                model="llama-cpp-qwen-9b",
            ),
            "Qwen/Qwen3.5-9B",
        )


if __name__ == "__main__":
    unittest.main()
