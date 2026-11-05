#!/usr/bin/env python3
"""Run TileLang performance regression benchmarks with optional manifest and timeout.

Default behaviour (no --manifest) reproduces the existing regression_all.py
auto-discovery so that upstream workflows remain fully compatible.

New options added for the PPU perf CI:
  --manifest FILE   Only run the benchmarks listed in FILE (repo-relative paths).
  --timeout SECS    Per-file wall-clock timeout in seconds (0 = unlimited).
  --repo-root DIR   Repo root for path resolution (default: auto-detect).
  --output FILE     Write structured JSON payload to FILE.

Exit codes:
  0  All listed files ran successfully (some may report latency of 0).
  1  At least one file failed or was not found.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from tabulate import tabulate
except Exception:  # pragma: no cover
    tabulate = None  # type: ignore

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

_RESULTS_JSON_PREFIX = "__TILELANG_PERF_RESULTS_JSON__="

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1


@dataclass
class CaseResult:
    name: str
    latency: float  # ms, NaN for failures
    status: str  # "success" | "failed" | "timeout" | "not_found"
    file: str  # backward-compatible alias for source
    source: str = ""  # stable repo-relative benchmark file identity
    error: str = ""
    duration_s: float = 0.0


@dataclass
class Payload:
    schema_version: int = SCHEMA_VERSION
    cases: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cases": self.cases,
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# Parsing helpers (reuse logic from regression_all.py)
# ---------------------------------------------------------------------------


def _parse_json_output(output: str) -> dict[str, float] | None:
    """Extract structured JSON result from benchmark stdout.

    Returns ``None`` when no ``__TILELANG_PERF_RESULTS_JSON__`` marker is
    found at all.  Returns an empty dict when the marker is present but the
    JSON payload is an empty list or empty dict (upstream semantics: marker
    presence means JSON path was taken, so text fallback must be suppressed).
    """
    for line in reversed(output.splitlines()):
        if line.startswith(_RESULTS_JSON_PREFIX):
            payload = line[len(_RESULTS_JSON_PREFIX) :].strip()
            items = json.loads(payload)
            if isinstance(items, dict):
                return {str(k): float(v) for k, v in items.items()}
            if isinstance(items, list) and not items:
                return {}
            data: dict[str, float] = {}
            for item in items:
                data[str(item["name"]).strip()] = float(item["latency"])
            return data
    return None


def _parse_text_output(output: str) -> dict[str, float]:
    """Fallback text parsing for legacy benchmarks."""
    data: dict[str, float] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, _, val = line.partition(":")
        name = name.strip()
        val = val.strip()
        if not name:
            continue
        try:
            data[name] = float(val)
        except ValueError:
            continue
    return data


def parse_output(output: str) -> dict[str, float]:
    """Parse benchmark stdout, preferring JSON then falling back to text.

    Text fallback is only used when no ``__TILELANG_PERF_RESULTS_JSON__``
    marker is present.  An empty JSON payload (empty list/dict) is *not*
    treated as absent — it returns ``{}`` so that noisy text lines (e.g.
    CUDA device capability) are never mis-parsed as latency values.
    """
    result = _parse_json_output(output)
    if result is not None:
        return result
    return _parse_text_output(output)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _repo_root_default() -> Path:
    """Auto-detect repo root (parent of maint/)."""
    return Path(__file__).resolve().parents[2]


def _examples_root(repo_root: Path) -> Path:
    return repo_root / "examples"


def discover_bench_files(examples_root: Path) -> list[Path]:
    """Auto-discover regression_*.py files under examples/."""
    files: list[Path] = []
    for pat in ("regression_*.py",):
        files.extend(examples_root.rglob(pat))
    return sorted({p for p in files if p.is_file() and p.name != "__init__.py"})


# ---------------------------------------------------------------------------
# Manifest loading (YAML schema v1)
# ---------------------------------------------------------------------------

_MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_PLATFORM = "ppu-890p"
_MANIFEST_KEYS = {"schema_version", "platform", "benchmarks", "excluded"}
_BENCHMARK_KEYS = {"path", "group", "expected_cases"}
_EXCLUDED_KEYS = {"path", "reason"}


def _validate_manifest_path(
    path_value: Any,
    location: str,
    repo_root: Path,
) -> tuple[str, Path]:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{location}: 'path' must be a non-empty string")

    path_str = path_value.strip()
    relative_path = Path(path_str)
    if relative_path.is_absolute():
        raise ValueError(f"{location}: path must be repo-relative: {path_str}")

    candidate = (repo_root / relative_path).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        raise ValueError(f"{location}: path escapes repo root: {path_str}") from None

    if not candidate.name.startswith("regression_") or not candidate.name.endswith(".py"):
        raise ValueError(f"{location}: does not match regression_*.py: {path_str}")
    return path_str, candidate


def _reject_unknown_keys(entry: dict[Any, Any], allowed: set[str], location: str) -> None:
    unknown = set(entry) - allowed
    if unknown:
        fields = ", ".join(sorted(repr(key) for key in unknown))
        raise ValueError(f"{location}: unknown field(s): {fields}")


def load_manifest(manifest_path: str, repo_root: Path) -> list[Path]:
    """Load a strict YAML v1 manifest and return enabled benchmark paths only."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML manifests; install it with 'pip install pyyaml'")

    manifest = Path(manifest_path)
    if manifest.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Manifest must be a YAML file (.yaml or .yml): {manifest}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest}")

    try:
        document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Manifest YAML parse error: {exc}") from exc

    if not isinstance(document, dict):
        raise ValueError(f"Manifest top-level must be a mapping, got {type(document).__name__}")
    _reject_unknown_keys(document, _MANIFEST_KEYS, "manifest")
    missing = _MANIFEST_KEYS - set(document)
    if missing:
        raise ValueError(f"Manifest missing required field(s): {', '.join(sorted(missing))}")

    schema_version = document["schema_version"]
    if type(schema_version) is not int or schema_version != _MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema_version: {schema_version!r} (expected {_MANIFEST_SCHEMA_VERSION})")
    if document["platform"] != _MANIFEST_PLATFORM:
        raise ValueError(f"Unsupported manifest platform: {document['platform']!r} (expected {_MANIFEST_PLATFORM!r})")

    benchmarks = document["benchmarks"]
    if not isinstance(benchmarks, list):
        raise ValueError(f"'benchmarks' must be a list, got {type(benchmarks).__name__}")
    if not benchmarks:
        raise ValueError("Manifest 'benchmarks' list must not be empty")

    excluded = document["excluded"]
    if not isinstance(excluded, list):
        raise ValueError(f"'excluded' must be a list, got {type(excluded).__name__}")

    seen: dict[str, str] = {}
    paths: list[Path] = []
    for index, entry in enumerate(benchmarks):
        location = f"benchmarks[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{location}: expected a mapping, got {type(entry).__name__}")
        _reject_unknown_keys(entry, _BENCHMARK_KEYS, location)
        path_str, candidate = _validate_manifest_path(entry.get("path"), location, repo_root)

        if "group" in entry and (not isinstance(entry["group"], str) or not entry["group"].strip()):
            raise ValueError(f"{location}: 'group' must be a non-empty string")
        if "expected_cases" in entry and (type(entry["expected_cases"]) is not int or entry["expected_cases"] < 1):
            raise ValueError(f"{location}: 'expected_cases' must be a positive integer")

        duplicate_key = str(candidate)
        if duplicate_key in seen:
            raise ValueError(f"{location}: duplicate path {path_str!r}; already used by {seen[duplicate_key]}")
        seen[duplicate_key] = location
        if not candidate.is_file():
            raise FileNotFoundError(f"{location}: file not found: {path_str}")
        paths.append(candidate)

    for index, entry in enumerate(excluded):
        location = f"excluded[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{location}: expected a mapping, got {type(entry).__name__}")
        _reject_unknown_keys(entry, _EXCLUDED_KEYS, location)
        path_str, candidate = _validate_manifest_path(entry.get("path"), location, repo_root)
        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{location}: 'reason' must be a non-empty string")

        duplicate_key = str(candidate)
        if duplicate_key in seen:
            raise ValueError(f"{location}: duplicate path {path_str!r}; already used by {seen[duplicate_key]}")
        seen[duplicate_key] = location

    return paths


# ---------------------------------------------------------------------------
# Single file runner
# ---------------------------------------------------------------------------


def run_single_benchmark(
    bench_file: Path,
    repo_root: Path,
    timeout: int = 0,
) -> list[CaseResult]:
    """Run one benchmark with a wall-clock timeout and drained output pipes."""
    repo_root_resolved = repo_root.resolve()
    bench_file_resolved = bench_file.resolve()
    try:
        source = str(bench_file_resolved.relative_to(repo_root_resolved))
    except ValueError:
        source = str(bench_file)

    if not bench_file_resolved.is_file():
        return [
            CaseResult(
                name=source,
                latency=float("nan"),
                status="not_found",
                file=source,
                source=source,
                error=f"Benchmark file not found: {bench_file}",
            )
        ]

    env = {**os.environ, "TL_PERF_REGRESSION_FORMAT": "json"}
    env["TILELANG_DISABLE_CACHE"] = "1"

    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            [sys.executable, str(bench_file_resolved)],
            cwd=str(bench_file_resolved.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        try:
            stdout_content, stderr_content = proc.communicate(
                timeout=timeout if timeout > 0 else None,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_content, stderr_content = proc.communicate()
            duration = time.monotonic() - t0
            for line in stdout_content.splitlines(keepends=True):
                if not line.startswith(_RESULTS_JSON_PREFIX):
                    print(line, end="", flush=True)
            details = stderr_content[-2000:] if stderr_content else ""
            return [
                CaseResult(
                    name=source,
                    latency=float("nan"),
                    status="timeout",
                    file=source,
                    source=source,
                    error=f"Timed out after {timeout}s\n{details}".rstrip(),
                    duration_s=duration,
                )
            ]

        duration = time.monotonic() - t0
        for line in stdout_content.splitlines(keepends=True):
            if not line.startswith(_RESULTS_JSON_PREFIX):
                print(line, end="", flush=True)

        if proc.returncode != 0:
            return [
                CaseResult(
                    name=source,
                    latency=float("nan"),
                    status="failed",
                    file=source,
                    source=source,
                    error=(f"Exit code {proc.returncode}\nSTDOUT:\n{stdout_content[-2000:]}\nSTDERR:\n{stderr_content[-2000:]}"),
                    duration_s=duration,
                )
            ]

        parsed = parse_output(stdout_content)
        if parsed is None:
            return [
                CaseResult(
                    name=source,
                    latency=float("nan"),
                    status="failed",
                    file=source,
                    source=source,
                    error=(f"No parseable benchmark output\nSTDERR:\n{stderr_content[-2000:]}"),
                    duration_s=duration,
                )
            ]

        return [
            CaseResult(
                name=name,
                latency=latency,
                status="success",
                file=source,
                source=source,
                duration_s=duration,
            )
            for name, latency in parsed.items()
        ]

    except Exception as exc:
        duration = time.monotonic() - t0
        return [
            CaseResult(
                name=source,
                latency=float("nan"),
                status="failed",
                file=source,
                source=source,
                error=str(exc)[:2000],
                duration_s=duration,
            )
        ]


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def build_payload(all_results: list[CaseResult], meta: dict[str, Any] | None = None) -> Payload:
    """Build structured payload from case results."""
    payload = Payload(meta=meta or {})
    for r in all_results:
        d = asdict(r)
        d["source"] = d["source"] or d["file"]
        # Convert NaN to None for JSON serialization
        if isinstance(d["latency"], float) and math.isnan(d["latency"]):
            d["latency"] = None
        payload.cases.append(d)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run TileLang performance regression benchmarks.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to YAML allowlist file (schema_version 1). If omitted, auto-discovers regression_*.py under examples/.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-file wall-clock timeout in seconds (0 = unlimited).",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root directory (default: auto-detect).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write structured JSON payload to this file.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root else _repo_root_default()
    repo_root = repo_root.resolve()
    if not repo_root.is_dir():
        print(f"ERROR: repo root not found: {repo_root}", file=sys.stderr)
        return 1

    # Determine file list
    if args.manifest:
        try:
            bench_files = load_manifest(args.manifest, repo_root)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"ERROR: manifest: {exc}", file=sys.stderr)
            return 1
    else:
        examples = _examples_root(repo_root)
        if not examples.is_dir():
            print(f"ERROR: examples dir not found: {examples}", file=sys.stderr)
            return 1
        bench_files = discover_bench_files(examples)
        if not bench_files:
            print(f"ERROR: no regression_*.py found under {examples}", file=sys.stderr)
            return 1

    total = len(bench_files)
    print(f"\n{'=' * 60}")
    print("  TileLang Performance Regression Suite")
    print(f"  Found {total} benchmark file(s)")
    if args.timeout > 0:
        print(f"  Per-file timeout: {args.timeout}s")
    print(f"{'=' * 60}")

    all_results: list[CaseResult] = []
    failures = 0

    for idx, bench_file in enumerate(bench_files, 1):
        rel_path = bench_file.relative_to(repo_root)
        print(f"\n{'─' * 60}")
        print(f"[{idx}/{total}] {rel_path}")
        print(f"{'─' * 60}")

        results = run_single_benchmark(bench_file, repo_root, timeout=args.timeout)
        all_results.extend(results)

        if any(r.status != "success" for r in results):
            failures += 1
        for r in results:
            if r.status == "success":
                print(f"  -> {r.name}: {r.latency:.4f} ms")
            elif r.status == "timeout":
                print(f"  -> TIMEOUT ({args.timeout}s)")
            elif r.status == "not_found":
                print(f"  -> NOT FOUND: {r.error[:200]}")
            elif r.status == "failed":
                print(f"  -> FAILED: {r.error[:200]}")

    # Summary
    successes = sum(1 for r in all_results if r.status == "success")
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"  Success cases: {successes}")
    print(f"  Failed/timeout files: {failures}")
    print(f"  Total cases: {len(all_results)}")

    # Build payload
    payload = build_payload(all_results)

    # Write structured output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(payload.to_dict(), f, indent=2)
        print(f"\n  Payload written to: {args.output}")

    # Also emit JSON line for backward compat with test_perf_regression.py
    merged = {}
    for r in all_results:
        if r.status == "success" and r.name not in merged:
            merged[r.name] = r.latency
    print(_RESULTS_JSON_PREFIX + json.dumps(merged, separators=(",", ":")))

    # Any manifest file failure is non-zero, while the full payload remains available.
    if failures:
        print(f"\nERROR: {failures} benchmark file(s) failed.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
