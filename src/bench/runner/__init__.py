from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from src.bench.composition import resolve_runtimes
from src.bench.runner.execution import run_runtime_matrix
from src.bench.runner.naming import experiment_group, timestamp_slug
from src.bench.runner.stats import (
    avg,
    ci95_half_width_for_rate,
    row_retrieval_exact,
    row_retrieval_score,
    row_throughput,
    row_total_time,
)
from src.bench.runner.summary import display_path, write_summary_reports
from src.bench.utils import default_output_filename, default_summary_stem, resolve_experiment_paths

ROOT_DIR = Path(__file__).resolve().parents[3]
LOG_DIR = ROOT_DIR / "logs"


def _resolve_runtimes_for_args(args: Any) -> list[str]:
    return resolve_runtimes(
        runtime_arg=str(getattr(args, "runtime", "auto") or "auto"),
        model_arg=getattr(args, "model", None),
        all_models=bool(getattr(args, "all_models", False)),
    )


def run_benchmark(args: Any) -> int:
    runtimes = _resolve_runtimes_for_args(args)

    exp_group = experiment_group(args)
    exp_stamp = timestamp_slug()
    experiment_dir, results_dir = resolve_experiment_paths(
        root_dir=ROOT_DIR,
        experiment_group=exp_group,
        experiment_stamp=exp_stamp,
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Managed mode: MLX servers run sequentially (never together) to avoid OOM.", file=sys.stderr)
    print(f"Artifacts directory: {experiment_dir}", file=sys.stderr)
    print(f"Results directory: {results_dir}", file=sys.stderr)

    results, setup_metrics = run_runtime_matrix(
        args=args,
        runtimes=runtimes,
        root_dir=ROOT_DIR,
        log_dir=LOG_DIR,
        experiment_dir=experiment_dir,
    )

    if args.output:
        output_name = args.output
    else:
        output_name = str(results_dir / default_output_filename(args, runtimes, results))

    output_arg = Path(output_name)
    if not output_arg.is_absolute() and output_arg.parent == Path("."):
        output_arg = Path("results") / output_arg

    output_path = output_arg if output_arg.is_absolute() else (ROOT_DIR / output_arg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result) + "\n")

    print(f"\n✅ Results written to {output_path}", file=sys.stderr)

    summary_stem = default_summary_stem(args, runtimes, results)
    summary_json_path, summary_md_path = write_summary_reports(
        args=args,
        results=results,
        runtimes=runtimes,
        setup_metrics=setup_metrics,
        output_path=output_path,
        artifact_dir=experiment_dir,
        root_dir=ROOT_DIR,
        summary_dir=results_dir,
        summary_stem=summary_stem,
    )
    print(f"✅ Summary JSON: {display_path(summary_json_path, ROOT_DIR)}", file=sys.stderr)
    print(f"✅ Summary MD: {display_path(summary_md_path, ROOT_DIR)}", file=sys.stderr)

    successful = [r for r in results if r.get("success")]
    if successful:
        print("\n📊 Summary:", file=sys.stderr)
        for runtime in runtimes:
            runtime_results = [r for r in successful if r.get("runtime") == runtime]
            if not runtime_results:
                continue
            avg_speed = avg([row_throughput(r, "tokens_per_second") for r in runtime_results])
            avg_time = avg([row_total_time(r) for r in runtime_results])
            retrieval_rows = [r for r in runtime_results if row_retrieval_score(r) is not None]
            avg_retrieval = avg([float(row_retrieval_score(r) or 0.0) for r in retrieval_rows])
            exact_rate = avg([1.0 if bool(row_retrieval_exact(r)) else 0.0 for r in retrieval_rows])
            exact_ci95 = ci95_half_width_for_rate(exact_rate, len(retrieval_rows))
            print(
                f"  {runtime}: avg {avg_speed:.2f} tok/s, avg {avg_time:.2f}s, "
                f"retrieval {avg_retrieval:.3f}, exact {exact_rate:.3f} +/- {exact_ci95:.3f}",
                file=sys.stderr,
            )

    if not results:
        setup_errors = [
            str(row.get("setup_error", "")).strip()
            for row in setup_metrics
            if str(row.get("setup_error", "")).strip()
        ]
        if setup_errors:
            print(
                "No benchmark runs were executed due setup/composition errors.",
                file=sys.stderr,
            )
            return 2

    return 0


__all__ = ["run_benchmark"]
