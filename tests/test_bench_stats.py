from __future__ import annotations

import unittest

from src.bench.runner.stats import row_prefill_sec
from src.bench.runner.summary import runtime_summary_rows


class BenchStatsTest(unittest.TestCase):
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
