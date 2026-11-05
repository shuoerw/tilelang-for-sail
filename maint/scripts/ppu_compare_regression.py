#!/usr/bin/env python3
"""Compare old vs new performance regression payloads.

Reads two structured JSON payloads (schema version 1) and produces:
  - Markdown report with comparison table
  - Machine-readable JSON with gate verdict
  - Optional PNG chart

Fail-closed rules:
  - Either payload has ALL cases failed/timeout -> correctness failure
  - Zero common successful cases -> correctness failure
  - Old success but new failed/timeout -> correctness failure
  - Latency is not a finite positive number -> correctness failure
  - Report generation failure -> correctness failure
  - Only-old / only-new cases -> reported, not regression
  - Performance regression -> reported, NOT a failure in Phase 0

Exit codes:
  0  No correctness failures (performance regressions reported but not blocking)
  1  Correctness failure detected
  2  Usage / schema error
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def load_payload(path: str) -> dict[str, Any]:
    """Load and validate a payload JSON file."""
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Payload is not a dict: {path}")

    sv = data.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version={sv} in {path}, expected {SCHEMA_VERSION}")

    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"'cases' is not a list in {path}")

    return data


def extract_cases(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract cases keyed by name. Returns {name: case_dict}."""
    result: dict[str, dict[str, Any]] = {}
    for case in payload["cases"]:
        name = case.get("name", "")
        if name and name not in result:
            result[name] = case
    return result


def _is_valid_latency(v: Any) -> bool:
    """Check that latency is a finite positive number."""
    if v is None:
        return False
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(fv) and fv > 0


def _case_source(case: dict[str, Any]) -> str:
    """Return stable source identity, accepting schema-v1 payloads without source."""
    return str(case.get("source") or case.get("file") or "")


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_payloads(
    old_payload: dict[str, Any],
    new_payload: dict[str, Any],
    regression_threshold: float = 0.95,
) -> dict[str, Any]:
    """Compare old and new payloads. Returns structured result dict."""

    old_cases = extract_cases(old_payload)
    new_cases = extract_cases(new_payload)

    # Categorize
    old_success = {k: v for k, v in old_cases.items() if v.get("status") == "success"}
    new_success = {k: v for k, v in new_cases.items() if v.get("status") == "success"}

    all_old_failed = len(old_success) == 0 and len(old_cases) > 0
    all_new_failed = len(new_success) == 0 and len(new_cases) > 0

    common_success = sorted(set(old_success) & set(new_success))
    only_old = sorted(set(old_cases) - set(new_cases))
    only_new = sorted(set(new_cases) - set(old_cases))

    # Old success but new failed/timeout under the same case name.
    old_ok_new_fail = sorted(k for k in old_success if k in new_cases and new_cases[k].get("status") != "success")

    # A file-level failure uses the source path as its stable identity. If the
    # old payload has successful cases from that source, every such case is a
    # correctness failure even though the failure record has a different name.
    old_success_by_source: dict[str, list[str]] = {}
    for name, case in old_success.items():
        source = _case_source(case)
        if source:
            old_success_by_source.setdefault(source, []).append(name)

    new_failures_by_source: dict[str, list[dict[str, Any]]] = {}
    for case in new_payload["cases"]:
        if case.get("status") not in {"failed", "timeout", "not_found"}:
            continue
        source = _case_source(case)
        if source:
            new_failures_by_source.setdefault(source, []).append(case)

    source_failures: list[dict[str, Any]] = []
    for source in sorted(set(old_success_by_source) & set(new_failures_by_source)):
        failed_cases = new_failures_by_source[source]
        source_failures.append(
            {
                "source": source,
                "new_statuses": sorted({str(case.get("status")) for case in failed_cases}),
                "affected_old_cases": sorted(old_success_by_source[source]),
                "errors": [str(case.get("error", "")) for case in failed_cases if case.get("error")],
            }
        )

    # Build comparison rows
    comparisons: list[dict[str, Any]] = []
    invalid_latency_cases: list[str] = []
    regressions: list[str] = []

    for name in common_success:
        old_lat = old_success[name].get("latency")
        new_lat = new_success[name].get("latency")

        if not _is_valid_latency(old_lat) or not _is_valid_latency(new_lat):
            invalid_latency_cases.append(name)
            continue

        old_lat_f = float(old_lat)
        new_lat_f = float(new_lat)
        speedup = old_lat_f / new_lat_f
        delta_pct = (speedup - 1.0) * 100.0

        status = "improved" if speedup >= 1.0 else "regressed"
        if speedup < regression_threshold:
            regressions.append(name)

        comparisons.append(
            {
                "name": name,
                "source": _case_source(old_success[name]) or _case_source(new_success[name]),
                "old_latency": old_lat_f,
                "new_latency": new_lat_f,
                "speedup": round(speedup, 6),
                "delta_pct": round(delta_pct, 2),
                "status": status,
            }
        )

    # Sort by speedup ascending (worst first)
    comparisons.sort(key=lambda x: x["speedup"])

    # Determine gate
    correctness_failures: list[str] = []

    if all_old_failed:
        correctness_failures.append("all_old_failed")
    if all_new_failed:
        correctness_failures.append("all_new_failed")
    if not common_success and not all_old_failed and not all_new_failed:
        correctness_failures.append("zero_common_success")
    if old_ok_new_fail:
        correctness_failures.append("old_success_new_failed")
    if source_failures:
        correctness_failures.append("source_failure")
    if invalid_latency_cases:
        correctness_failures.append("invalid_latency")

    gate = {
        "correctness_ok": len(correctness_failures) == 0,
        "correctness_failures": correctness_failures,
        "source_failures": source_failures,
        "performance_regressions": regressions,
        "regression_threshold": regression_threshold,
        "has_performance_regression": len(regressions) > 0,
        "num_compared": len(comparisons),
        "num_regressions": len(regressions),
    }

    result = {
        "comparisons": comparisons,
        "only_old": only_old,
        "only_new": only_new,
        "old_success_new_failed": old_ok_new_fail,
        "source_failures": source_failures,
        "invalid_latency_cases": invalid_latency_cases,
        "gate": gate,
        "summary": {
            "old_total": len(old_cases),
            "old_success": len(old_success),
            "new_total": len(new_cases),
            "new_success": len(new_success),
            "common_success": len(common_success),
        },
    }

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_markdown(result: dict[str, Any]) -> str:
    """Generate a Markdown report from comparison result."""
    lines: list[str] = []

    gate = result["gate"]
    summary = result["summary"]

    lines.append("## Performance Regression Report")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Old cases (success/total) | {summary['old_success']}/{summary['old_total']} |")
    lines.append(f"| New cases (success/total) | {summary['new_success']}/{summary['new_total']} |")
    lines.append(f"| Common success | {summary['common_success']} |")
    lines.append(f"| Regressions (< {gate['regression_threshold']:.2f}x) | {gate['num_regressions']} |")
    lines.append(f"| Correctness OK | {'Yes' if gate['correctness_ok'] else 'NO'} |")
    lines.append("")

    if result["comparisons"]:
        lines.append("### Comparison")
        lines.append("")
        lines.append("| Benchmark | Old (ms) | New (ms) | Speedup | Delta | Status |")
        lines.append("|-----------|----------|----------|---------|-------|--------|")
        for c in result["comparisons"]:
            lines.append(
                f"| {c['name']} | {c['old_latency']:.4f} | {c['new_latency']:.4f} "
                f"| {c['speedup']:.4f}x | {c['delta_pct']:+.2f}% | {c['status']} |"
            )
        lines.append("")

    if result["old_success_new_failed"]:
        lines.append("### Old Success -> New Failed")
        lines.append("")
        for name in result["old_success_new_failed"]:
            lines.append(f"- {name}")
        lines.append("")

    if result["source_failures"]:
        lines.append("### Source-level Failures")
        lines.append("")
        for failure in result["source_failures"]:
            statuses = ", ".join(failure["new_statuses"])
            affected = ", ".join(failure["affected_old_cases"])
            lines.append(f"- `{failure['source']}`: new status **{statuses}**; old successful cases: {affected}")
        lines.append("")

    if result["only_old"]:
        lines.append("### Only in Old (not in New)")
        lines.append("")
        for name in result["only_old"]:
            lines.append(f"- {name}")
        lines.append("")

    if result["only_new"]:
        lines.append("### Only in New (not in Old)")
        lines.append("")
        for name in result["only_new"]:
            lines.append(f"- {name}")
        lines.append("")

    if result["invalid_latency_cases"]:
        lines.append("### Invalid Latency")
        lines.append("")
        for name in result["invalid_latency_cases"]:
            lines.append(f"- {name}")
        lines.append("")

    if gate["correctness_failures"]:
        lines.append("### Correctness Failures")
        lines.append("")
        for f in gate["correctness_failures"]:
            lines.append(f"- {f}")
        lines.append("")

    verdict = "PASS" if gate["correctness_ok"] else "FAIL (correctness)"
    if gate["correctness_ok"] and gate["has_performance_regression"]:
        verdict = "WARN (performance regression, Phase 0 report-only)"

    lines.append(f"**Verdict: {verdict}**")
    lines.append("")

    return "\n".join(lines)


def generate_chart(result: dict[str, Any], output_path: str) -> None:
    """Generate a PNG chart of speedups."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("WARNING: matplotlib not available, skipping chart generation", file=sys.stderr)
        return

    comparisons = result["comparisons"]
    if not comparisons:
        return

    names = [c["name"] for c in comparisons]
    speedups = [c["speedup"] for c in comparisons]

    n = len(names)
    fig_h = min(max(6.0, 0.35 * n + 2.2), 22.0)
    fig, ax = plt.subplots(figsize=(14.0, fig_h))

    colors = ["#2ecc71" if s >= 1.0 else "#e74c3c" for s in speedups]
    y = np.arange(n)

    ax.barh(y, speedups, color=colors, edgecolor="black", linewidth=0.4, height=0.72)
    ax.axvline(1.0, linestyle="--", linewidth=1.4, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Speedup (old / new)")
    ax.set_title("Performance Regression Analysis")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, s in enumerate(speedups):
        delta = (s - 1.0) * 100.0
        label = f"{s:.3f}x ({delta:+.1f}%)"
        ha = "left" if s >= 1.0 else "right"
        offset = 0.003 if s >= 1.0 else -0.003
        ax.text(s + offset, i, label, va="center", ha=ha, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare perf regression payloads.")
    parser.add_argument("--old", required=True, help="Path to old (baseline) payload JSON.")
    parser.add_argument("--new", required=True, help="Path to new (PR) payload JSON.")
    parser.add_argument("--output-md", default=None, help="Write Markdown report to file.")
    parser.add_argument("--output-json", default=None, help="Write machine-readable JSON to file.")
    parser.add_argument("--output-png", default=None, help="Write comparison chart PNG.")
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.95,
        help="Speedup below this is flagged as regression (default: 0.95).",
    )
    args = parser.parse_args()

    # Load payloads
    try:
        old_payload = load_payload(args.old)
        new_payload = load_payload(args.new)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Compare
    try:
        result = compare_payloads(old_payload, new_payload, args.regression_threshold)
    except Exception as exc:
        print(f"ERROR during comparison: {exc}", file=sys.stderr)
        return 2

    # Generate report
    try:
        md = generate_markdown(result)
    except Exception as exc:
        print(f"ERROR generating report: {exc}", file=sys.stderr)
        return 1

    print(md)

    if args.output_md:
        Path(args.output_md).write_text(md)
        print(f"\nMarkdown written to: {args.output_md}", file=sys.stderr)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"JSON written to: {args.output_json}", file=sys.stderr)

    if args.output_png:
        generate_chart(result, args.output_png)
        print(f"Chart written to: {args.output_png}", file=sys.stderr)

    # Emit gate fields to stdout for machine consumption
    gate = result["gate"]
    print("\n--- GATE ---", file=sys.stderr)
    print(f"correctness_ok={gate['correctness_ok']}", file=sys.stderr)
    print(f"has_performance_regression={gate['has_performance_regression']}", file=sys.stderr)
    print(f"num_regressions={gate['num_regressions']}", file=sys.stderr)

    # Exit code: only correctness failures block
    if not gate["correctness_ok"]:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
