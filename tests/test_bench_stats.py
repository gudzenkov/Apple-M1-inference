from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
