#!/usr/bin/env python3
"""Tests for PPU performance regression CI scripts.

Covers:
  - run_current_regression.py: manifest loading, discovery, timeout, payload structure
  - compare_perf_regression.py: comparison logic, fail-closed rules, gate verdicts
  - workflow YAML: structural assertions
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# Add maint/scripts to path so we can import the modules
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "maint" / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import ppu_run_regression as rcr
import ppu_compare_regression as cpr


# ===========================================================================
# Helpers
# ===========================================================================


def _make_payload(cases: list[dict], schema_version: int = 1, meta: dict | None = None) -> dict:
    """Build a minimal valid payload dict."""
    return {
        "schema_version": schema_version,
        "cases": cases,
        "meta": meta or {},
    }


def _write_payload(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def _make_case(name: str, latency: float | None, status: str = "success", file: str = "", error: str = "", source: str = "") -> dict:
    file_identity = file or source or name
    case = {
        "name": name,
        "latency": latency,
        "status": status,
        "file": file_identity,
        "error": error,
        "duration_s": 1.0,
    }
    if source:
        case["source"] = source
    return case


# ===========================================================================
# Tests for run_current_regression.py — manifest loading
# ===========================================================================


class TestManifestLoading:
    """Test strict YAML manifest parsing and validation."""

    @staticmethod
    def _repo_with_benchmark(tmp_path: Path) -> tuple[Path, Path]:
        repo = tmp_path / "repo"
        bench = repo / "examples" / "ppu" / "topk" / "regression_topk.py"
        bench.parent.mkdir(parents=True)
        bench.write_text("# benchmark")
        return repo, bench

    @staticmethod
    def _write_manifest(tmp_path: Path, document: object, suffix: str = ".yaml") -> Path:
        manifest = tmp_path / f"manifest{suffix}"
        manifest.write_text(yaml.safe_dump(document, sort_keys=False))
        return manifest

    @staticmethod
    def _valid_document(path: str = "examples/ppu/topk/regression_topk.py") -> dict:
        return {
            "schema_version": 1,
            "platform": "ppu-890p",
            "benchmarks": [{"path": path}],
            "excluded": [],
        }

    def test_valid_manifest(self, tmp_path: Path):
        repo, bench = self._repo_with_benchmark(tmp_path)
        document = self._valid_document()
        document["benchmarks"][0].update({"group": "ppu", "expected_cases": 1})
        manifest = self._write_manifest(tmp_path, document)

        assert rcr.load_manifest(str(manifest), repo) == [bench.resolve()]

    def test_wrong_schema_version_fails(self, tmp_path: Path):
        repo, _ = self._repo_with_benchmark(tmp_path)
        document = self._valid_document()
        document["schema_version"] = 2
        manifest = self._write_manifest(tmp_path, document)

        with pytest.raises(ValueError, match="schema_version"):
            rcr.load_manifest(str(manifest), repo)

    def test_wrong_platform_fails(self, tmp_path: Path):
        repo, _ = self._repo_with_benchmark(tmp_path)
        document = self._valid_document()
        document["platform"] = "ppu-other"
        manifest = self._write_manifest(tmp_path, document)

        with pytest.raises(ValueError, match="platform"):
            rcr.load_manifest(str(manifest), repo)

    @pytest.mark.parametrize("benchmarks", [None, {}, "path", 1])
    def test_benchmarks_must_be_a_list(self, tmp_path: Path, benchmarks: object):
        repo, _ = self._repo_with_benchmark(tmp_path)
        document = self._valid_document()
        document["benchmarks"] = benchmarks
        manifest = self._write_manifest(tmp_path, document)

        with pytest.raises(ValueError, match="'benchmarks' must be a list"):
            rcr.load_manifest(str(manifest), repo)

    def test_benchmark_entry_must_be_a_mapping(self, tmp_path: Path):
        repo, _ = self._repo_with_benchmark(tmp_path)
        document = self._valid_document()
        document["benchmarks"] = ["examples/ppu/topk/regression_topk.py"]
        manifest = self._write_manifest(tmp_path, document)

        with pytest.raises(ValueError, match="expected a mapping"):
            rcr.load_manifest(str(manifest), repo)

    def test_benchmark_path_must_be_nonempty(self, tmp_path: Path):
        repo, _ = self._repo_with_benchmark(tmp_path)
        document = self._valid_document()
        document["benchmarks"] = [{"path": ""}]
        manifest = self._write_manifest(tmp_path, document)

        with pytest.raises(ValueError, match="non-empty string"):
            rcr.load_manifest(str(manifest), repo)

    def test_unknown_benchmark_field_fails(self, tmp_path: Path):
        repo, _ = self._repo_with_benchmark(tmp_path)
        document = self._valid_document()
        document["benchmarks"][0]["note"] = "not in schema"
        manifest = self._write_manifest(tmp_path, document)

        with pytest.raises(ValueError, match="unknown field"):
            rcr.load_manifest(str(manifest), repo)

    @pytest.mark.parametrize("expected_cases", [0, -1, True, "1", []])
    def test_expected_cases_must_be_a_positive_integer(self, tmp_path: Path, expected_cases: object):
        repo, _ = self._repo_with_benchmark(tmp_path)
        document = self._valid_document()
        document["benchmarks"][0]["expected_cases"] = expected_cases
        manifest = self._write_manifest(tmp_path, document)

        with pytest.raises(ValueError, match="positive integer"):
            rcr.load_manifest(str(manifest), repo)

    def test_duplicate_path_fails(self, tmp_path: Path):
        repo, _ = self._repo_with_benchmark(tmp_path)
        document = self._valid_document()
        document["benchmarks"] *= 2
        manifest = self._write_manifest(tmp_path, document)

        with pytest.raises(ValueError, match="duplicate path"):
            rcr.load_manifest(str(manifest), repo)

    def test_normalized_duplicate_path_fails(self, tmp_path: Path):
        repo, _ = self._repo_with_benchmark(tmp_path)
        document = self._valid_document()
        document["benchmarks"].append(
            {
                "path": "examples/ppu/topk/../topk/regression_topk.py",
            }
        )
        manifest = self._write_manifest(tmp_path, document)

        with pytest.raises(ValueError, match="duplicate path"):
            rcr.load_manifest(str(manifest), repo)

    def test_excluded_entry_is_validated_but_not_executed(self, tmp_path: Path):
        repo, bench = self._repo_with_benchmark(tmp_path)
        document = self._valid_document()
        document["excluded"] = [
            {
                "path": "examples/ppu/missing/regression_excluded.py",
                "reason": "Known unsupported operation",
            }
        ]
        manifest = self._write_manifest(tmp_path, document)

        assert rcr.load_manifest(str(manifest), repo) == [bench.resolve()]

    def test_excluded_reason_must_be_nonempty(self, tmp_path: Path):
        repo, _ = self._repo_with_benchmark(tmp_path)
        document = self._valid_document()
        document["excluded"] = [
            {
                "path": "examples/ppu/missing/regression_excluded.py",
                "reason": "",
            }
        ]
        manifest = self._write_manifest(tmp_path, document)

        with pytest.raises(ValueError, match="reason.*non-empty string"):
            rcr.load_manifest(str(manifest), repo)

    def test_txt_manifest_is_rejected(self, tmp_path: Path):
        repo, _ = self._repo_with_benchmark(tmp_path)
        manifest = self._write_manifest(tmp_path, self._valid_document(), suffix=".txt")

        with pytest.raises(ValueError, match="YAML file"):
            rcr.load_manifest(str(manifest), repo)

    def test_empty_benchmarks_fails(self, tmp_path: Path):
        repo, _ = self._repo_with_benchmark(tmp_path)
        document = self._valid_document()
        document["benchmarks"] = []
        manifest = self._write_manifest(tmp_path, document)

        with pytest.raises(ValueError, match="must not be empty"):
            rcr.load_manifest(str(manifest), repo)

    def test_path_escape_fails(self, tmp_path: Path):
        repo, _ = self._repo_with_benchmark(tmp_path)
        manifest = self._write_manifest(tmp_path, self._valid_document("../../etc/regression_evil.py"))

        with pytest.raises(ValueError, match="escapes repo root"):
            rcr.load_manifest(str(manifest), repo)

    def test_missing_enabled_file_fails(self, tmp_path: Path):
        repo, _ = self._repo_with_benchmark(tmp_path)
        manifest = self._write_manifest(
            tmp_path,
            self._valid_document("examples/ppu/missing/regression_missing.py"),
        )

        with pytest.raises(FileNotFoundError, match="file not found"):
            rcr.load_manifest(str(manifest), repo)

    def test_non_regression_file_fails(self, tmp_path: Path):
        repo, _ = self._repo_with_benchmark(tmp_path)
        manifest = self._write_manifest(tmp_path, self._valid_document("examples/ppu/topk/benchmark_topk.py"))

        with pytest.raises(ValueError, match="regression_"):
            rcr.load_manifest(str(manifest), repo)

    def test_manifest_file_not_found(self, tmp_path: Path):
        repo, _ = self._repo_with_benchmark(tmp_path)
        with pytest.raises(FileNotFoundError, match="Manifest file not found"):
            rcr.load_manifest(str(tmp_path / "no_such.yaml"), repo)


class TestDiscovery:
    """Test default auto-discovery (no manifest)."""

    def test_discover_finds_regression_files(self, tmp_path: Path):
        """Auto-discovery finds regression_*.py files."""
        examples = tmp_path / "examples"
        (examples / "ppu" / "a").mkdir(parents=True)
        (examples / "ppu" / "a" / "regression_a.py").write_text("# a")
        (examples / "ppu" / "a" / "other.py").write_text("# other")

        result = rcr.discover_bench_files(examples)
        assert len(result) == 1
        assert result[0].name == "regression_a.py"


# ===========================================================================
# Tests for run_current_regression.py — payload structure
# ===========================================================================


class TestPayloadStructure:
    """Test payload building."""

    def test_payload_schema_version(self):
        """Payload has schema_version=1."""
        results = [rcr.CaseResult("t", 1.5, "success", "f.py")]
        payload = rcr.build_payload(results)
        assert payload.schema_version == 1

    def test_nan_latency_becomes_none(self):
        """NaN latency is serialized as None."""
        results = [rcr.CaseResult("t", float("nan"), "failed", "f.py", error="boom")]
        payload = rcr.build_payload(results)
        d = payload.to_dict()
        assert d["cases"][0]["latency"] is None

    def test_timeout_in_payload(self):
        """Timeout status preserved in payload."""
        results = [rcr.CaseResult("t", float("nan"), "timeout", "f.py", error="timed out")]
        payload = rcr.build_payload(results)
        d = payload.to_dict()
        assert d["cases"][0]["status"] == "timeout"


class TestParseOutput:
    """Test output parsing."""

    def test_json_prefix_parsing(self):
        """JSON prefix line is parsed correctly."""
        output = 'some log\n__TILELANG_PERF_RESULTS_JSON__={"a":1.5,"b":2.0}\n'
        result = rcr.parse_output(output)
        assert result == {"a": 1.5, "b": 2.0}

    def test_json_list_format(self):
        """JSON list format is parsed correctly."""
        data = [{"name": "x", "latency": 3.0}]
        output = f"__TILELANG_PERF_RESULTS_JSON__={json.dumps(data)}\n"
        result = rcr.parse_output(output)
        assert result == {"x": 3.0}

    def test_text_fallback(self):
        """Text key:value fallback works."""
        output = "bench_a: 1.5\nbench_b: 2.0\n"
        result = rcr.parse_output(output)
        assert result == {"bench_a": 1.5, "bench_b": 2.0}

    def test_no_marker_returns_none_from_json_parser(self):
        """_parse_json_output returns None when no marker is present."""
        assert rcr._parse_json_output("some noise\nno marker here\n") is None

    def test_empty_list_marker_returns_empty_dict(self):
        """Empty list JSON marker returns {} (not None), suppressing text fallback."""
        output = "__TILELANG_PERF_RESULTS_JSON__=[]\n"
        assert rcr._parse_json_output(output) == {}

    def test_empty_dict_marker_returns_empty_dict(self):
        """Empty dict JSON marker returns {} (not None), suppressing text fallback."""
        output = "__TILELANG_PERF_RESULTS_JSON__={}\n"
        assert rcr._parse_json_output(output) == {}

    def test_empty_list_marker_with_capability_noise_no_fake_latency(self):
        """Empty list marker + CUDA capability noise must NOT produce fake latency.

        This is the core regression scenario: the benchmark emits an empty
        JSON result followed by device-probe lines like
        ``CUDA device capability: 89``.  The old truthiness check caused
        fallback to text parsing which picked up 89 as a latency value.
        """
        output = "Loading PPU driver...\n__TILELANG_PERF_RESULTS_JSON__=[]\nCUDA device capability: 89\nDevice 0: Some PPU Card\n"
        result = rcr.parse_output(output)
        assert result == {}
        # Crucially, 89 must NOT appear as a latency value
        assert 89.0 not in result.values()

    def test_empty_dict_marker_with_capability_noise_no_fake_latency(self):
        """Empty dict marker + noise lines also returns empty dict."""
        output = "__TILELANG_PERF_RESULTS_JSON__={}\nCUDA device capability: 89\n"
        result = rcr.parse_output(output)
        assert result == {}


class TestRunSingleBenchmark:
    """Exercise the real subprocess path, including pipe and timeout behavior."""

    @staticmethod
    def _script(tmp_path: Path, name: str, body: str) -> tuple[Path, Path]:
        repo = tmp_path / "repo"
        bench_dir = repo / "examples" / "ppu" / "test"
        bench_dir.mkdir(parents=True, exist_ok=True)
        script = bench_dir / name
        script.write_text(body)
        return repo, script

    def test_success_json_subprocess(self, tmp_path: Path):
        repo, script = self._script(
            tmp_path,
            "regression_success.py",
            "print('__TILELANG_PERF_RESULTS_JSON__={\"case_a\":1.25}')\n",
        )
        results = rcr.run_single_benchmark(script, repo, timeout=2)
        assert [(r.name, r.status, r.latency) for r in results] == [("case_a", "success", 1.25)]
        assert results[0].source == "examples/ppu/test/regression_success.py"

    def test_nonzero_subprocess_preserves_traceback(self, tmp_path: Path):
        repo, script = self._script(
            tmp_path,
            "regression_failure.py",
            "import sys\nprint('before failure')\nprint('traceback marker', file=sys.stderr)\nraise SystemExit(3)\n",
        )
        result = rcr.run_single_benchmark(script, repo, timeout=2)[0]
        assert result.status == "failed"
        assert "Exit code 3" in result.error
        assert "before failure" in result.error
        assert "traceback marker" in result.error

    def test_large_stderr_does_not_deadlock(self, tmp_path: Path):
        repo, script = self._script(
            tmp_path,
            "regression_stderr.py",
            "import sys\nsys.stderr.write('x' * 200000)\nprint('__TILELANG_PERF_RESULTS_JSON__={\"case_a\":2.5}')\n",
        )
        result = rcr.run_single_benchmark(script, repo, timeout=3)[0]
        assert result.status == "success"
        assert result.latency == 2.5

    def test_sleep_subprocess_times_out(self, tmp_path: Path):
        repo, script = self._script(
            tmp_path,
            "regression_timeout.py",
            "import time\nprint('started', flush=True)\ntime.sleep(5)\n",
        )
        result = rcr.run_single_benchmark(script, repo, timeout=1)[0]
        assert result.status == "timeout"
        assert "Timed out after 1s" in result.error
        assert result.duration_s < 4

    def test_missing_subprocess_is_not_found(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        missing = repo / "examples" / "regression_missing.py"
        result = rcr.run_single_benchmark(missing, repo, timeout=1)[0]
        assert result.status == "not_found"
        assert result.source == "examples/regression_missing.py"

    def test_partial_manifest_writes_payload_and_exits_nonzero(self, tmp_path: Path):
        import subprocess

        repo = tmp_path / "repo"
        bench_dir = repo / "examples" / "ppu"
        bench_dir.mkdir(parents=True)
        good = bench_dir / "regression_good.py"
        bad = bench_dir / "regression_bad.py"
        good.write_text("print('__TILELANG_PERF_RESULTS_JSON__={\"good\":1.0}')\n")
        bad.write_text("raise RuntimeError('broken benchmark')\n")
        manifest = repo / "manifest.yaml"
        manifest.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "platform": "ppu-890p",
                    "benchmarks": [
                        {"path": "examples/ppu/regression_good.py"},
                        {"path": "examples/ppu/regression_bad.py"},
                    ],
                    "excluded": [],
                },
                sort_keys=False,
            )
        )
        output = repo / "payload.json"

        completed = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS_DIR / "ppu_run_regression.py"),
                "--manifest",
                str(manifest),
                "--repo-root",
                str(repo),
                "--timeout",
                "2",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert completed.returncode == 1
        payload = json.loads(output.read_text())
        assert {case["status"] for case in payload["cases"]} == {"success", "failed"}
        assert all(case["source"].startswith("examples/ppu/") for case in payload["cases"])

    def test_relative_repo_root_cli(self, tmp_path: Path):
        """CLI with --repo-root as relative path must not raise ValueError.

        Regression test: load_manifest resolves paths to absolute while a
        relative repo_root caused relative_to() to fail inside main().
        """
        import subprocess as sp

        repo = tmp_path / "repo"
        bench_dir = repo / "examples" / "ppu"
        bench_dir.mkdir(parents=True)
        bench = bench_dir / "regression_rel.py"
        bench.write_text("print('__TILELANG_PERF_RESULTS_JSON__={\"k\":1.0}')\n")
        manifest = repo / "manifest.yaml"
        manifest.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "platform": "ppu-890p",
                    "benchmarks": [{"path": "examples/ppu/regression_rel.py"}],
                    "excluded": [],
                },
                sort_keys=False,
            )
        )
        output = repo / "payload.json"

        completed = sp.run(
            [
                sys.executable,
                str(_SCRIPTS_DIR / "ppu_run_regression.py"),
                "--manifest",
                str(manifest),
                "--repo-root",
                ".",
                "--timeout",
                "5",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(repo),
        )
        assert completed.returncode == 0, f"Expected exit 0, got {completed.returncode}\nSTDERR: {completed.stderr[-2000:]}"
        payload = json.loads(output.read_text())
        assert payload["cases"][0]["status"] == "success"

    def test_empty_json_with_noise_no_fake_case(self, tmp_path: Path):
        """A benchmark emitting empty JSON marker + CUDA capability noise
        must NOT fabricate a case with latency 89.

        Empty JSON marker is a legitimate result (zero cases), not a parse
        failure.  The subprocess path must never fall through to text
        parsing which would pick up '89' from capability noise.
        """
        repo, script = self._script(
            tmp_path,
            "regression_empty.py",
            (
                "import sys\n"
                "print('Loading PPU driver...')\n"
                "print('__TILELANG_PERF_RESULTS_JSON__=[]')\n"
                "print('CUDA device capability: 89')\n"
            ),
        )
        results = rcr.run_single_benchmark(script, repo, timeout=5)
        # Empty JSON marker → zero cases, NOT a "failed" status
        assert results == []
        # Must not contain 89 anywhere as latency
        for r in results:
            if r.status == "success":
                assert r.latency != 89.0, "Fabricated case from CUDA capability noise"

    def test_empty_dict_marker_subprocess_returns_no_cases(self, tmp_path: Path):
        """Empty dict JSON marker via subprocess returns zero cases (not failure)."""
        repo, script = self._script(
            tmp_path,
            "regression_empty_dict.py",
            "print('__TILELANG_PERF_RESULTS_JSON__={}')\n",
        )
        results = rcr.run_single_benchmark(script, repo, timeout=5)
        assert results == []
        assert all(r.latency != 89.0 for r in results if r.status == "success")

    def test_relative_repo_root_subprocess(self, tmp_path: Path):
        """run_single_benchmark works when repo_root is a relative Path.

        Regression test for the ValueError when bench_file.relative_to()
        was called on an un-resolved relative repo_root.
        """
        repo, script = self._script(
            tmp_path,
            "regression_relroot.py",
            "print('__TILELANG_PERF_RESULTS_JSON__={\"case_rel\":0.5}')\n",
        )
        # Build a relative path to the repo directory
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            rel_repo = Path(os.path.relpath(repo, tmp_path))
            results = rcr.run_single_benchmark(script, rel_repo, timeout=5)
            assert len(results) == 1
            assert results[0].name == "case_rel"
            assert results[0].status == "success"
        finally:
            os.chdir(old_cwd)


# ===========================================================================
# Tests for compare_perf_regression.py
# ===========================================================================


class TestCompare:
    """Test comparison logic and fail-closed rules."""

    def test_improvement(self):
        """Speedup > 1 classified as improved."""
        old = _make_payload([_make_case("a", 2.0)])
        new = _make_payload([_make_case("a", 1.0)])
        result = cpr.compare_payloads(old, new)
        assert result["comparisons"][0]["status"] == "improved"
        assert result["comparisons"][0]["speedup"] == 2.0
        assert result["gate"]["correctness_ok"] is True

    def test_single_regression(self):
        """Speedup < threshold classified as regression."""
        old = _make_payload([_make_case("a", 1.0)])
        new = _make_payload([_make_case("a", 2.0)])  # new is slower
        result = cpr.compare_payloads(old, new, regression_threshold=0.95)
        assert result["comparisons"][0]["status"] == "regressed"
        assert result["gate"]["has_performance_regression"] is True
        # Phase 0: correctness is still OK
        assert result["gate"]["correctness_ok"] is True

    def test_performance_only_regression_cli_is_report_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        old = _write_payload(tmp_path, "old.json", _make_payload([_make_case("a", 1.0)]))
        new = _write_payload(tmp_path, "new.json", _make_payload([_make_case("a", 2.0)]))
        monkeypatch.setattr(sys, "argv", ["ppu_compare_regression.py", "--old", str(old), "--new", str(new)])

        assert cpr.main() == 0

    def test_two_regressions(self):
        """Multiple regressions are counted."""
        old = _make_payload([_make_case("a", 1.0), _make_case("b", 1.0)])
        new = _make_payload([_make_case("a", 2.0), _make_case("b", 3.0)])
        result = cpr.compare_payloads(old, new)
        assert result["gate"]["num_regressions"] == 2

    def test_zero_common_success(self):
        """Zero common success cases -> correctness failure."""
        old = _make_payload([_make_case("a", 1.0)])
        new = _make_payload([_make_case("b", 1.0)])
        result = cpr.compare_payloads(old, new)
        assert result["gate"]["correctness_ok"] is False
        assert "zero_common_success" in result["gate"]["correctness_failures"]

    def test_old_success_new_failed(self):
        """Old success but new failed -> correctness failure."""
        old = _make_payload([_make_case("a", 1.0)])
        new = _make_payload([_make_case("a", None, status="failed", error="crash")])
        result = cpr.compare_payloads(old, new)
        assert result["gate"]["correctness_ok"] is False
        assert "old_success_new_failed" in result["gate"]["correctness_failures"]

    def test_old_success_new_timeout(self):
        """Old success but new timeout -> correctness failure."""
        old = _make_payload([_make_case("a", 1.0)])
        new = _make_payload([_make_case("a", None, status="timeout")])
        result = cpr.compare_payloads(old, new)
        assert result["gate"]["correctness_ok"] is False
        assert "old_success_new_failed" in result["gate"]["correctness_failures"]

    def test_all_old_failed(self):
        """All old cases failed -> correctness failure."""
        old = _make_payload([_make_case("a", None, status="failed")])
        new = _make_payload([_make_case("a", 1.0)])
        result = cpr.compare_payloads(old, new)
        assert result["gate"]["correctness_ok"] is False
        assert "all_old_failed" in result["gate"]["correctness_failures"]

    def test_all_new_failed(self):
        """All new cases failed -> correctness failure."""
        old = _make_payload([_make_case("a", 1.0)])
        new = _make_payload([_make_case("a", None, status="failed")])
        result = cpr.compare_payloads(old, new)
        assert result["gate"]["correctness_ok"] is False

    def test_only_old_only_new_reported(self):
        """Only-old and only-new cases are reported but don't fail."""
        old = _make_payload([_make_case("shared", 1.0), _make_case("only_old", 1.0)])
        new = _make_payload([_make_case("shared", 1.0), _make_case("only_new", 1.0)])
        result = cpr.compare_payloads(old, new)
        assert result["only_old"] == ["only_old"]
        assert result["only_new"] == ["only_new"]
        assert result["gate"]["correctness_ok"] is True

    def test_invalid_latency_zero(self):
        """Latency of 0 is invalid."""
        old = _make_payload([_make_case("a", 0.0)])
        new = _make_payload([_make_case("a", 1.0)])
        result = cpr.compare_payloads(old, new)
        assert "invalid_latency" in result["gate"]["correctness_failures"]

    def test_invalid_latency_negative(self):
        """Negative latency is invalid."""
        old = _make_payload([_make_case("a", -1.0)])
        new = _make_payload([_make_case("a", 1.0)])
        result = cpr.compare_payloads(old, new)
        assert "invalid_latency" in result["gate"]["correctness_failures"]

    def test_multicase_source_failure_marks_all_old_cases(self):
        source = "examples/ppu/topk/regression_topk.py"
        old = _make_payload(
            [
                _make_case("topk_a", 1.0, source=source),
                _make_case("topk_b", 2.0, source=source),
            ]
        )
        new = _make_payload(
            [
                _make_case(source, None, status="timeout", source=source, error="timeout"),
            ]
        )
        result = cpr.compare_payloads(old, new)
        assert result["gate"]["correctness_ok"] is False
        assert "source_failure" in result["gate"]["correctness_failures"]
        assert result["source_failures"][0]["affected_old_cases"] == ["topk_a", "topk_b"]

    def test_source_failure_blocks_when_other_source_succeeds(self):
        failed_source = "examples/ppu/topk/regression_topk.py"
        good_source = "examples/ppu/flash/regression_flash.py"
        old = _make_payload(
            [
                _make_case("topk_a", 1.0, source=failed_source),
                _make_case("flash_a", 2.0, source=good_source),
            ]
        )
        new = _make_payload(
            [
                _make_case(failed_source, None, status="failed", source=failed_source),
                _make_case("flash_a", 1.9, source=good_source),
            ]
        )
        result = cpr.compare_payloads(old, new)
        assert result["summary"]["common_success"] == 1
        assert result["gate"]["correctness_ok"] is False
        assert result["gate"]["source_failures"][0]["source"] == failed_source
        assert "Source-level Failures" in cpr.generate_markdown(result)

    def test_schema_error(self, tmp_path: Path):
        """Wrong schema version raises ValueError."""
        bad = _write_payload(tmp_path, "bad.json", {"schema_version": 99, "cases": []})
        with pytest.raises(ValueError, match="schema_version"):
            cpr.load_payload(str(bad))


class TestCompareMarkdown:
    """Test Markdown report generation."""

    def test_markdown_contains_verdict(self):
        """Report contains a verdict line."""
        old = _make_payload([_make_case("a", 1.0)])
        new = _make_payload([_make_case("a", 1.0)])
        result = cpr.compare_payloads(old, new)
        md = cpr.generate_markdown(result)
        assert "**Verdict:" in md

    def test_markdown_fail_verdict(self):
        """Correctness failure shows FAIL in verdict."""
        old = _make_payload([_make_case("a", 1.0)])
        new = _make_payload([_make_case("a", None, status="failed")])
        result = cpr.compare_payloads(old, new)
        md = cpr.generate_markdown(result)
        assert "FAIL" in md


# ===========================================================================
# Tests for CLI help (smoke tests)
# ===========================================================================


class TestCLIHelp:
    """Verify CLI --help works without error."""

    def test_run_current_regression_help(self):
        import subprocess

        r = subprocess.run(
            [sys.executable, str(_SCRIPTS_DIR / "ppu_run_regression.py"), "--help"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0
        assert "--manifest" in r.stdout
        assert "--timeout" in r.stdout

    def test_compare_perf_regression_help(self):
        import subprocess

        r = subprocess.run(
            [sys.executable, str(_SCRIPTS_DIR / "ppu_compare_regression.py"), "--help"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0
        assert "--old" in r.stdout
        assert "--new" in r.stdout


# ===========================================================================
# Tests for workflow YAML structure
# ===========================================================================


class TestWorkflowYAML:
    """Lightweight structural tests on the workflow YAML."""

    @pytest.fixture(autouse=True)
    def load_workflow(self):
        import yaml

        wf_path = _REPO_ROOT / ".github" / "workflows" / "ppu-perf-regression-bot.yml"
        with open(wf_path) as f:
            self.wf = yaml.safe_load(f)

    @staticmethod
    def _write_publisher_artifact(artifact_dir: Path, mode: str) -> None:
        artifact_dir.mkdir()
        benchmark_name = {
            "marker-json": "<!-- ppu-perf-regression-890p kind=comment request_order=999 run_id=999 run_attempt=1 -->",
            "html-json": "<details><summary>forged</summary>payload</details>",
            "mention-json": "@everyone forged mention",
        }.get(mode, "基准测试")
        source = "examples/ppu/基准测试.py"
        old_payload = {
            "schema_version": 1,
            "cases": [
                {
                    "name": benchmark_name,
                    "latency": 10.0,
                    "status": "success",
                    "file": source,
                    "source": source,
                    "error": "",
                    "duration_s": 1.0,
                }
            ],
            "meta": {},
        }
        new_payload = {
            "schema_version": 1,
            "cases": [
                {
                    "name": benchmark_name,
                    "latency": 8.0,
                    "status": "success",
                    "file": source,
                    "source": source,
                    "error": "",
                    "duration_s": 1.0,
                }
            ],
            "meta": {},
        }
        comparison = {
            "comparisons": [
                {
                    "name": benchmark_name,
                    "source": source,
                    "old_latency": 10.0,
                    "new_latency": 8.0,
                    "speedup": 1.25,
                    "delta_pct": 25.0,
                    "status": "improved",
                }
            ],
            "only_old": [],
            "only_new": [],
            "old_success_new_failed": [],
            "source_failures": [],
            "invalid_latency_cases": [],
            "gate": {
                "correctness_ok": True,
                "correctness_failures": [],
                "source_failures": [],
                "performance_regressions": [],
                "regression_threshold": 0.95,
                "has_performance_regression": False,
                "num_compared": 1,
                "num_regressions": 0,
            },
            "summary": {
                "old_total": 1,
                "old_success": 1,
                "new_total": 1,
                "new_success": 1,
                "common_success": 1,
            },
        }
        if mode == "summary-mismatch":
            comparison["summary"]["common_success"] = 0
        report_injection = {
            "valid": "仅存在于artifact的Unicode文本",
            "hash-mismatch": "hash mismatch",
            "mutated-bytes": "mutated bytes",
            "mutated-report-bytes": "report bytes before trusted digest",
            "metadata-schema-v2": "legacy metadata schema",
            "summary-mismatch": "forged summary",
            "extra-file": "extra file",
            "marker-report": "<!-- ppu-perf-regression-890p kind=comment request_order=999 run_id=999 run_attempt=1 -->",
            "html-report": "<details><summary>forged</summary>payload</details>",
            "mention-report": "@everyone forged mention",
            "long-report": "A" * 56000,
            "invalid-utf8-report": "invalid UTF-8 placeholder",
            "marker-json": "structured marker fixture",
            "html-json": "structured HTML fixture",
            "mention-json": "structured mention fixture",
        }[mode]
        report = f"## Performance Regression Report\n\n{report_injection}\n\n**Verdict: PASS**\n".encode()
        if mode == "invalid-utf8-report":
            report = b"## Performance Regression Report\n\n\xff\n\n**Verdict: PASS**\n"
        png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x00\x00\x00\x00"
        files = {
            "gate_rc.txt": b"0\n",
            "regression_result.md": report,
            "regression_result.png": png,
            "regression_compare.json": (json.dumps(comparison, sort_keys=True) + "\n").encode(),
            "old_payload.json": (json.dumps(old_payload, sort_keys=True) + "\n").encode(),
            "new_payload.json": (json.dumps(new_payload, sort_keys=True) + "\n").encode(),
        }
        for name, data in files.items():
            (artifact_dir / name).write_bytes(data)
        metadata = {
            "schema_version": 3,
            "pr_number": 17,
            "request_kind": "comment",
            "request_order": 101,
            "comment_id": 101,
            "base_ref": "main",
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "new_tree": "c" * 40,
            "run_id": "201",
            "run_attempt": 1,
            "gate_rc": "0",
            "artifact_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        }
        if mode == "hash-mismatch":
            metadata["artifact_sha256"]["regression_compare.json"] = "0" * 64
        if mode == "metadata-schema-v2":
            metadata["schema_version"] = 2
        (artifact_dir / "publisher_metadata.json").write_text(json.dumps(metadata, sort_keys=True) + "\n")
        if mode == "mutated-bytes":
            with (artifact_dir / "new_payload.json").open("ab") as output:
                output.write(b" ")
        if mode == "mutated-report-bytes":
            with (artifact_dir / "regression_result.md").open("ab") as output:
                output.write(b"\npost-metadata mutation\n")
        if mode == "extra-file":
            (artifact_dir / "unexpected.txt").write_text("unexpected")

    def _run_publisher(
        self,
        tmp_path: Path,
        *,
        lint_result: str,
        ppu_result: str,
        new_tree: str,
        artifact_mode: str = "missing",
        newer_owner: bool = False,
    ) -> dict:
        artifact_dir = tmp_path / "publisher-artifact"
        if artifact_mode == "malicious":
            artifact_dir.mkdir()
            (artifact_dir / "ATTACKER_CONTROLLED_TEXT").write_text("$(touch should-not-run)")
        elif artifact_mode != "missing":
            self._write_publisher_artifact(artifact_dir, artifact_mode)

        publisher_env = {
            "PR_NUMBER": "17",
            "BASE_REF": "main",
            "BASE_SHA": "a" * 40,
            "HEAD_SHA": "b" * 40,
            "COMMENT_ID": "101",
            "REQUEST_KIND": "comment",
            "REQUEST_ORDER": "101",
            "REQUEST_KEY": "comment-101",
            "TRIGGER_LABEL": "@ppu-regression (comment #101)",
            "STATUS_COMMENT_ID": "500",
            "EXPECTED_NEW_TREE": new_tree,
            "LINT_RESULT": lint_result,
            "PPU_RESULT": ppu_result,
            "ARTIFACT_DIR": str(artifact_dir),
            "GITHUB_WORKSPACE": str(tmp_path),
            "GITHUB_RUN_ID": "201",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_REPOSITORY": "t-head/tilelang-for-sail",
            "GITHUB_SERVER_URL": "https://github.com",
            "PPU_IMAGE": ("download.t-head.cn/docker_release/pytorch:sdk2.2.0-2.10.0-ubuntu24.04-hggcrt3-py312-20261005"),
        }
        publish = next(
            step for step in self.wf["jobs"]["publish-result"]["steps"] if step["name"] == "Validate artifact and publish result"
        )
        script = publish["with"]["script"]
        owned_request_order = 102 if newer_owner else 101
        owned_body = f"<!-- ppu-perf-regression-890p kind=comment request_order={owned_request_order} run_id=201 run_attempt=1 -->\nRunning"

        node = shutil.which("node")
        electron_node = Path("/Applications/Qoder.app/Contents/MacOS/Qoder")
        electron_mode = node is None and electron_node.is_file()
        if electron_mode:
            node = str(electron_node)
        runtime_prefix = ""
        if node is None:
            osascript = shutil.which("osascript")
            if osascript is None:
                pytest.skip("Node.js or JavaScript for Automation is required")
            runtime_prefix = (
                f"const process = {{ env: {json.dumps(publisher_env)}, exitCode: 0 }};\n"
                "const Buffer = {\n"
                "  byteLength: value => unescape(encodeURIComponent(String(value))).length,\n"
                "  alloc: () => { throw new Error('unexpected Buffer.alloc'); },\n"
                "  from: () => { throw new Error('unexpected Buffer.from'); },\n"
                "};\n"
                "function require(name) {\n"
                "  if (name === 'path') return { basename: value => String(value).split('/').pop(), join: (...parts) => parts.join('/').replace(/\\/{2,}/g, '/') };\n"
                "  if (name === 'fs') return { constants: { O_RDONLY: 0, O_NOFOLLOW: 0 }, lstatSync: () => { throw new Error('ATTACKER_CONTROLLED_TEXT $(touch should-not-run)'); } };\n"
                "  if (name === 'crypto') return {};\n"
                "  throw new Error(`unexpected module: ${name}`);\n"
                "}\n"
            )

        harness = (
            runtime_prefix + "const state = { failures: [], notices: [], gets: 0, updates: 0, body: null };\n"
            "const core = { setFailed: value => state.failures.push(String(value)), notice: value => state.notices.push(String(value)) };\n"
            "const context = { repo: { owner: 't-head', repo: 'tilelang-for-sail' } };\n"
            f"const ownedBody = {json.dumps(owned_body)};\n"
            "const github = { rest: { issues: {\n"
            "  getComment: async () => { state.gets += 1; return { data: { user: { login: 'github-actions[bot]', type: 'Bot' }, body: ownedBody } }; },\n"
            "  updateComment: async args => { state.updates += 1; state.body = args.body; },\n"
            "} } };\n"
            "const emit = () => console.log('PUBLISHER_STATE=' + JSON.stringify(state));\n"
            "(async () => {\n" + script + "\n})().then(emit, error => { state.crash = String(error); process.exitCode = 1; emit(); });\n"
        )
        if node is not None:
            command = [node, "-e", harness]
        else:
            harness_path = tmp_path / "publisher-simulation.js"
            harness_path.write_text(harness)
            command = [osascript, "-l", "JavaScript", str(harness_path)]
        runtime_env = dict(os.environ, **publisher_env)
        if electron_mode:
            runtime_env["ELECTRON_RUN_AS_NODE"] = "1"
        result = subprocess.run(command, capture_output=True, text=True, env=runtime_env, timeout=30)
        combined_output = f"{result.stdout}\n{result.stderr}"
        marker = "PUBLISHER_STATE="
        state_lines = [line.split(marker, 1)[1] for line in combined_output.splitlines() if marker in line]
        assert result.returncode == 0 and state_lines, combined_output
        state = json.loads(state_lines[-1])
        assert "crash" not in state, state
        return state

    def test_triggers(self):
        """Workflow triggers on @ppu-regression comments and curated dev pushes."""
        # PyYAML parses bare 'on:' as boolean True key
        on = self.wf.get("on") or self.wf.get(True)
        assert on is not None, "No 'on' trigger block found"
        assert set(on) == {"issue_comment", "push"}
        assert on["issue_comment"] == {"types": ["created"]}
        assert on["push"] == {"branches": ["feature/ci-testing-dev"]}
        assert "pull_request" not in on
        assert "workflow_dispatch" not in on

    def test_permissions_are_deny_by_default_and_job_scoped(self):
        """Only serialized status jobs may write issues; execution is read-only."""
        assert self.wf["permissions"] == {}
        jobs = self.wf["jobs"]
        assert jobs["resolve-pr"]["permissions"] == {"contents": "read"}
        assert jobs["claim-request"]["permissions"] == {"issues": "write"}
        assert jobs["pre-commit-lint"]["permissions"] == {"contents": "read"}
        assert jobs["ppu-perf"]["permissions"] == {"contents": "read"}
        assert jobs["publish-result"]["permissions"] == {"issues": "write"}
        assert "pull-requests" not in str(self.wf)

    def test_no_workflow_level_concurrency(self):
        """Unauthorized comments cannot cancel authorized work."""
        assert "concurrency" not in self.wf

    def test_job_level_concurrency(self):
        """Execution supersedes, while all trusted status writes are serialized."""
        for job_name in ("pre-commit-lint", "ppu-perf"):
            conc = self.wf["jobs"][job_name]["concurrency"]
            assert "ppu-perf-890p-pr-" in conc["group"]
            assert "needs.resolve-pr.outputs.pr_number" in conc["group"]
            assert conc["cancel-in-progress"] is True
        status_groups = []
        for job_name in ("claim-request", "publish-result"):
            conc = self.wf["jobs"][job_name]["concurrency"]
            status_groups.append(conc["group"])
            assert "needs.resolve-pr.outputs.pr_number" in conc["group"]
            assert conc["cancel-in-progress"] is False
        assert len(set(status_groups)) == 1
        assert "ppu-perf-890p-status-pr-" in status_groups[0]

    def test_exact_job_set_and_controller_contract(self):
        jobs = self.wf["jobs"]
        assert set(jobs) == {
            "resolve-pr",
            "claim-request",
            "pre-commit-lint",
            "ppu-perf",
            "publish-result",
        }
        resolve = jobs["resolve-pr"]
        assert resolve["runs-on"] == "ubuntu-latest"
        resolve_if = str(resolve["if"])
        assert "@ppu-regression" in resolve_if
        assert "github.event_name == 'issue_comment'" in resolve_if
        assert "github.event_name == 'push'" in resolve_if
        assert "refs/heads/feature/ci-testing-dev" in resolve_if
        assert set(resolve["outputs"]) == {
            "pr_number",
            "base_ref",
            "base_sha",
            "head_sha",
            "comment_id",
            "request_kind",
            "request_order",
            "request_key",
            "trigger_label",
        }
        claim = jobs["claim-request"]
        assert claim["needs"] == "resolve-pr"
        assert set(claim["outputs"]) == {"should_run", "status_comment_id"}

    def test_resolver_uses_api_and_strict_guards(self):
        """PR identity, dynamic base, and full SHAs are frozen from the API."""
        resolve_text = str(self.wf["jobs"]["resolve-pr"])
        for required in (
            "getCollaboratorPermissionLevel",
            "pulls.get",
            "pr.base.ref",
            "pr.base.sha",
            "pr.head.sha",
            "pr.state !== 'open'",
            "pr.merged",
            "Fork PR not supported",
            "^[0-9a-f]{40}$",
        ):
            assert required in resolve_text
        # The resolver may reference github.sha / github.ref only inside the push
        # branch (via context.sha / context.payload.ref), but never for checkout.
        assert "actions/checkout" not in resolve_text
        assert "refs/pull/" not in resolve_text

    def test_claim_uses_strict_fixed_marker_and_frozen_outputs(self):
        claim = self.wf["jobs"]["claim-request"]
        step = next(step for step in claim["steps"] if step["name"] == "Claim status comment")
        script = step["with"]["script"]
        for key, output in (
            ("PR_NUMBER", "pr_number"),
            ("BASE_REF", "base_ref"),
            ("BASE_SHA", "base_sha"),
            ("HEAD_SHA", "head_sha"),
            ("COMMENT_ID", "comment_id"),
            ("REQUEST_KIND", "request_kind"),
            ("REQUEST_ORDER", "request_order"),
            ("REQUEST_KEY", "request_key"),
            ("TRIGGER_LABEL", "trigger_label"),
        ):
            assert f"needs.resolve-pr.outputs.{output}" in claim["env"][key]
        assert script.startswith("core.setOutput('should_run', 'false');")
        assert "kind=(comment|push)" in script
        assert "request_order=([1-9][0-9]*)" in script
        assert "run_id=([1-9][0-9]*)" in script
        assert "run_attempt=([1-9][0-9]*)" in script
        assert "Number.isSafeInteger" in script
        assert "kindNamespacePattern" in script and "slotPattern.test(comment.body)" in script
        assert "github-actions[bot]" in script and "comment.user?.type === 'Bot'" in script
        assert "body?.includes" not in script and ".sort((a, b)" not in script
        assert "updateComment" in script and "createComment" in script

    def test_delayed_old_resolver_cannot_overwrite_new_request(self):
        script = self.wf["jobs"]["claim-request"]["steps"][0]["with"]["script"]
        stale_guard = "currentOwnership?.requestOrder > requestOrder"
        assert stale_guard in script
        assert "A higher request order already owns the status comment; skipping" in script
        assert script.index(stale_guard) < script.index("github.rest.issues.updateComment")
        assert "needs.claim-request.outputs.should_run == 'true'" in str(self.wf["jobs"]["pre-commit-lint"]["if"])
        assert "needs.claim-request.outputs.should_run == 'true'" in str(self.wf["jobs"]["ppu-perf"]["if"])

    def test_same_request_rerun_has_monotonic_attempt_ownership(self):
        script = self.wf["jobs"]["claim-request"]["steps"][0]["with"]["script"]
        assert "currentOwnership?.requestOrder === requestOrder" in script
        assert "currentOwnership.runId !== runId" in script
        assert "same request order is associated with a different run ID" in script
        assert "currentOwnership.runAttempt > runAttempt" in script
        assert "A higher run attempt already owns this request; skipping" in script
        assert script.index("currentOwnership.runAttempt > runAttempt") < script.index("github.rest.issues.updateComment")

    def test_malformed_conflicting_and_nonbot_markers_fail_closed(self):
        script = self.wf["jobs"]["claim-request"]["steps"][0]["with"]["script"]
        assert "isTrustedBot(comment)" in script
        assert "slotPattern.test(comment.body)" in script
        assert "status marker does not match the fixed ownership format" in script
        assert "Malformed bot status marker" in script
        assert "Multiple bot status markers found; refusing ambiguous ownership" in script
        assert "comment.user?.login === 'github-actions[bot]'" in script
        assert "comment.user?.type === 'Bot'" in script
        assert "core.setOutput('should_run', 'true')" in script
        # Cross-kind marker in the same slot must also fail closed.
        assert "status marker kind mismatch" in script

    def test_execution_jobs_have_no_secret_or_comment_write_path(self):
        for name in ("pre-commit-lint", "ppu-perf"):
            job_text = str(self.wf["jobs"][name])
            assert "secrets." not in job_text
            assert "issues: write" not in job_text
            assert "updateComment" not in job_text
            assert "createComment" not in job_text
            assert "github-script" not in job_text

    def test_execution_checkouts_are_frozen_and_noncredentialed(self):
        workflow_text = (_REPO_ROOT / ".github" / "workflows" / "ppu-perf-regression-bot.yml").read_text()
        assert "refs/pull/" not in workflow_text
        for name in ("pre-commit-lint", "ppu-perf"):
            steps = self.wf["jobs"][name]["steps"]
            checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout@"))
            assert checkout["uses"] == "actions/checkout@v7"
            assert checkout["with"]["ref"] == "${{ needs.resolve-pr.outputs.head_sha }}"
            assert checkout["with"]["fetch-depth"] == 0
            assert checkout["with"]["submodules"] == "recursive"
            assert checkout["with"]["persist-credentials"] is False

    def test_both_execution_jobs_construct_the_frozen_merge(self):
        """OLD is base_sha; NEW is a verified local base/head merge tree."""
        commands = []
        for name in ("pre-commit-lint", "ppu-perf"):
            step = next(step for step in self.wf["jobs"][name]["steps"] if step["name"] == "Construct frozen merge snapshot")
            command = step["run"]
            commands.append(command)
            for required in (
                "git fetch --no-tags --force",
                'git checkout --detach "$BASE_SHA"',
                'merge --no-ff --no-edit "$HEAD_SHA"',
                'git rev-parse HEAD^1)" = "$BASE_SHA"',
                'git rev-parse HEAD^2)" = "$HEAD_SHA"',
                "git rev-parse 'HEAD^{tree}'",
                "git merge --abort",
                "Frozen base/head merge conflict",
            ):
                assert required in command
            assert "git checkout main" not in command
            assert "git checkout dev" not in command
        assert "steps.pr-info.outputs.base_ref" in self.wf["jobs"]["resolve-pr"]["outputs"]["base_ref"]
        assert 'test "$NEW_TREE" = "$EXPECTED_NEW_TREE"' in commands[1]
        assert "needs.pre-commit-lint.outputs.new_tree" in self.wf["jobs"]["ppu-perf"]["env"]["EXPECTED_NEW_TREE"]

    def test_pre_commit_lint_configuration_and_commands(self):
        job = self.wf["jobs"]["pre-commit-lint"]
        assert job["name"] == "Quick Lint"
        assert set(job["needs"]) == {"resolve-pr", "claim-request"}
        assert "needs.claim-request.outputs.should_run == 'true'" in str(job["if"])
        assert job["runs-on"] == "ubuntu-latest"
        assert job["timeout-minutes"] == 30
        steps = job["steps"]
        setup = next(step for step in steps if step.get("uses", "").startswith("actions/setup-python@"))
        assert setup["uses"] == "actions/setup-python@v6"
        assert setup["with"]["python-version"] == "3.10"
        runs = [step["run"] for step in steps if "run" in step]
        assert "python -m compileall -q -f tilelang" in runs
        assert "pipx run pre-commit run --all-files --color=always --show-diff-on-failure" in runs
        assert "continue-on-error" not in str(job)

    def test_ppu_perf_depends_on_resolver_claim_and_lint_outputs(self):
        job = self.wf["jobs"]["ppu-perf"]
        assert set(job["needs"]) == {"resolve-pr", "claim-request", "pre-commit-lint"}
        assert "needs.claim-request.outputs.should_run == 'true'" in str(job["if"])
        env = job["env"]
        for key, output in (
            ("PR_NUMBER", "pr_number"),
            ("BASE_REF", "base_ref"),
            ("BASE_SHA", "base_sha"),
            ("HEAD_SHA", "head_sha"),
            ("COMMENT_ID", "comment_id"),
            ("REQUEST_KIND", "request_kind"),
            ("REQUEST_ORDER", "request_order"),
            ("REQUEST_KEY", "request_key"),
            ("TRIGGER_LABEL", "trigger_label"),
        ):
            assert f"needs.resolve-pr.outputs.{output}" in env[key]
        assert "continue-on-error" not in job

    def test_scheduler_step_exists_only_in_ppu_perf(self):
        """Only the gated PPU job can start the hardware scheduler action."""
        scheduler_jobs = []
        for job_name, job in self.wf["jobs"].items():
            if any("ppu-scheduler-action" in step.get("uses", "") for step in job.get("steps", [])):
                scheduler_jobs.append(job_name)
        assert scheduler_jobs == ["ppu-perf"]

    def test_ppu_perf_timeout(self):
        """PPU job has 360-minute timeout."""
        assert self.wf["jobs"]["ppu-perf"]["timeout-minutes"] == 360

    def test_ppu_scheduler_action_is_reviewed_commit_pin(self):
        steps = self.wf["jobs"]["ppu-perf"]["steps"]
        scheduler = next(step for step in steps if step.get("uses", "").startswith("t-head/ppu-scheduler-action"))
        assert scheduler["uses"] == "t-head/ppu-scheduler-action@aac83e464f2a3fcbddddc60c8016021c996092a6"
        assert "ZW-M890P" in scheduler["with"]["node_selector"]
        assert scheduler["with"]["timeout_minutes"] == 300
        assert scheduler["with"]["image"] == "${{ env.PPU_IMAGE }}"

    def test_scheduler_receives_and_revalidates_frozen_snapshot(self):
        steps = self.wf["jobs"]["ppu-perf"]["steps"]
        scheduler = next(step for step in steps if "ppu-scheduler-action" in step.get("uses", ""))
        extra_env = scheduler["with"]["extra_env"]
        assert "\n" not in extra_env and ", " not in extra_env
        for name in (
            "BASE_SHA",
            "HEAD_SHA",
            "NEW_TREE",
            "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT",
            "REQUEST_KIND",
            "REQUEST_ORDER",
        ):
            assert f"{name}=" in extra_env
        command = scheduler["with"]["command"]
        assert 'git rev-parse HEAD^1)" = "$BASE_SHA"' in command
        assert 'git rev-parse HEAD^2)" = "$HEAD_SHA"' in command
        assert 'git rev-parse \'HEAD^{tree}\')" = "$NEW_TREE"' in command
        assert 'git checkout --detach "$BASE_SHA"' in command
        assert 'test "$(git rev-parse HEAD)" = "$BASE_SHA"' in command
        assert "git checkout main" not in command
        assert "refs/pull/" not in command

    def test_retry_isolation_and_exact_artifact_name(self):
        perf = self.wf["jobs"]["ppu-perf"]
        scheduler = next(step for step in perf["steps"] if "ppu-scheduler-action" in step.get("uses", ""))
        assert "${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" in scheduler["with"]["command"]
        upload = next(step for step in perf["steps"] if step.get("uses", "").startswith("actions/upload-artifact@"))
        expected = "ppu-perf-regression-890p-${{ github.run_id }}-${{ github.run_attempt }}"
        assert upload["with"]["name"] == expected
        assert upload["with"]["if-no-files-found"] == "error"
        publish = self.wf["jobs"]["publish-result"]
        download = next(step for step in publish["steps"] if step.get("uses", "").startswith("actions/download-artifact@"))
        assert download["with"]["name"] == expected
        assert "publisher_metadata.json" in upload["with"]["path"]
        assert "gate_rc.txt" in upload["with"]["path"]

    def test_publisher_is_isolated_from_pr_code(self):
        job = self.wf["jobs"]["publish-result"]
        assert set(job["needs"]) == {"resolve-pr", "claim-request", "pre-commit-lint", "ppu-perf"}
        assert "always()" in str(job["if"])
        assert "needs.resolve-pr.result == 'success'" in str(job["if"])
        assert "needs.claim-request.result == 'success'" in str(job["if"])
        assert "needs.claim-request.outputs.should_run == 'true'" in str(job["if"])
        assert job["runs-on"] == "ubuntu-latest"
        assert all("run" not in step for step in job["steps"])
        download = next(step for step in job["steps"] if step["name"] == "Download exact current-run artifact")
        publish = next(step for step in job["steps"] if step["name"] == "Validate artifact and publish result")
        assert "needs.ppu-perf.result == 'success'" in download["if"]
        assert "needs.ppu-perf.result == 'failure'" in download["if"]
        assert publish["if"] == "always()"
        assert "cancelled" in publish["with"]["script"]
        assert "required benchmark artifact is unavailable or invalid" in publish["with"]["script"]
        job_text = str(job)
        for forbidden in ("actions/checkout", "actions/setup-python", "ppu-scheduler-action", "secrets."):
            assert forbidden not in job_text
        for key, output in (
            ("PR_NUMBER", "pr_number"),
            ("BASE_REF", "base_ref"),
            ("BASE_SHA", "base_sha"),
            ("HEAD_SHA", "head_sha"),
            ("COMMENT_ID", "comment_id"),
            ("REQUEST_KIND", "request_kind"),
            ("REQUEST_ORDER", "request_order"),
            ("REQUEST_KEY", "request_key"),
            ("TRIGGER_LABEL", "trigger_label"),
        ):
            assert f"needs.resolve-pr.outputs.{output}" in job["env"][key]
        assert "needs.claim-request.outputs.status_comment_id" in job["env"]["STATUS_COMMENT_ID"]

    def test_publisher_strictly_validates_artifact_and_metadata(self):
        steps = self.wf["jobs"]["publish-result"]["steps"]
        publish = next(step for step in steps if step["name"] == "Validate artifact and publish result")
        script = publish["with"]["script"]
        for required in (
            "lstatSync",
            "isSymbolicLink",
            "maxBytes",
            "exactKeys",
            "publisher_metadata.json",
            "gate_rc.txt",
            "regression_compare.json",
            "regression_result.md",
            "metadata.schema_version !== 3",
            "metadata.pr_number !== prNumber",
            "metadata.request_kind !== requestKind",
            "metadata.request_order !== requestOrder",
            "metadata.comment_id !== expectedCommentField",
            "metadata.base_sha !== process.env.BASE_SHA",
            "metadata.head_sha !== process.env.HEAD_SHA",
            "metadata.new_tree !== process.env.EXPECTED_NEW_TREE",
            "metadata.run_id !== process.env.GITHUB_RUN_ID",
            "metadata.run_attempt !== runAttempt",
            "metadata.gate_rc !== gateRc",
            "exactKeys(metadata.artifact_sha256, hashedArtifactFiles",
            "crypto.createHash('sha256')",
            "crypto.timingSafeEqual",
            "typeof gate.correctness_ok !== 'boolean'",
            "typeof gate.has_performance_regression !== 'boolean'",
            "gate exit code disagrees",
            "required benchmark artifact is unavailable or invalid",
        ):
            assert required in script

    def test_schema_v3_metadata_writer_hashes_the_complete_publisher_artifact(self):
        writer = next(step for step in self.wf["jobs"]["ppu-perf"]["steps"] if step["name"] == "Write trusted publisher metadata")
        script = writer["run"]
        assert '"schema_version": 3' in script
        assert '"gate_rc": gate_rc[0]' in script
        assert '"artifact_sha256": {' in script
        assert "hashlib.sha256(data).hexdigest()" in script
        assert "os.O_RDONLY | directory | nofollow" in script
        for name in (
            "gate_rc.txt",
            "regression_result.md",
            "regression_result.png",
            "regression_compare.json",
            "old_payload.json",
            "new_payload.json",
        ):
            assert f'"{name}"' in script
        assert '"schema_version": 2' not in script

    def test_publisher_tree_validation_is_conditional_on_artifact_consumption(self):
        publish = next(
            step for step in self.wf["jobs"]["publish-result"]["steps"] if step["name"] == "Validate artifact and publish result"
        )
        script = publish["with"]["script"]
        unconditional_sha_check = "if (!shaPattern.test(process.env.BASE_SHA) ||\n      !shaPattern.test(process.env.HEAD_SHA))"
        conditional_tree_check = "if (!shaPattern.test(process.env.EXPECTED_NEW_TREE))"
        assert unconditional_sha_check in script
        assert conditional_tree_check in script
        assert script.index("const requiresArtifact") < script.index("if (requiresArtifact)")
        assert script.index("if (requiresArtifact)") < script.index(conditional_tree_check)
        assert "frozen base/head/tree SHA is invalid" not in script
        assert "validationError" not in script

    @pytest.mark.parametrize(
        ("lint_result", "ppu_result", "new_tree", "artifact_mode", "expected_status"),
        [
            pytest.param(
                "failure",
                "skipped",
                "",
                "missing",
                "Lint or frozen merge check failed — PPU benchmarks skipped",
                id="merge-failed-before-tree-output",
            ),
            pytest.param(
                "failure",
                "skipped",
                "c" * 40,
                "missing",
                "Lint or frozen merge check failed — PPU benchmarks skipped",
                id="lint-failure",
            ),
            pytest.param(
                "cancelled",
                "skipped",
                "",
                "missing",
                "Lint check cancelled, usually because a newer request superseded this run",
                id="lint-cancelled",
            ),
            pytest.param(
                "skipped",
                "skipped",
                "",
                "missing",
                "Lint check skipped — PPU benchmarks did not run",
                id="lint-skipped",
            ),
            pytest.param(
                "success",
                "cancelled",
                "",
                "missing",
                "PPU run cancelled, usually because a newer request superseded this run",
                id="ppu-cancelled",
            ),
            pytest.param(
                "success",
                "skipped",
                "",
                "missing",
                "PPU run skipped after lint completed",
                id="ppu-skipped",
            ),
            pytest.param(
                "success",
                "failure",
                "c" * 40,
                "missing",
                "Infrastructure failure: required benchmark artifact is unavailable or invalid",
                id="ppu-failure-artifact-missing",
            ),
            pytest.param(
                "success",
                "failure",
                "c" * 40,
                "malicious",
                "Infrastructure failure: required benchmark artifact is unavailable or invalid",
                id="ppu-failure-artifact-malicious",
            ),
            pytest.param(
                "success",
                "success",
                "",
                "missing",
                "Infrastructure failure: required benchmark artifact is unavailable or invalid",
                id="successful-ppu-path-missing-tree",
            ),
        ],
    )
    def test_publisher_failure_paths_update_owned_comment(
        self,
        tmp_path: Path,
        lint_result: str,
        ppu_result: str,
        new_tree: str,
        artifact_mode: str,
        expected_status: str,
    ):
        state = self._run_publisher(
            tmp_path,
            lint_result=lint_result,
            ppu_result=ppu_result,
            new_tree=new_tree,
            artifact_mode=artifact_mode,
        )
        assert state["failures"] == []
        assert state["gets"] == 1
        assert state["updates"] == 1
        assert expected_status in state["body"]
        assert "ATTACKER_CONTROLLED_TEXT" not in state["body"]
        assert "$(touch should-not-run)" not in state["body"]

    def test_failure_publisher_still_refuses_newer_comment_owner(self, tmp_path: Path):
        state = self._run_publisher(
            tmp_path,
            lint_result="success",
            ppu_result="success",
            new_tree="",
            newer_owner=True,
        )
        assert state["failures"] == []
        assert state["gets"] == 1
        assert state["updates"] == 0
        assert state["body"] is None
        assert state["notices"] == ["A newer request or run attempt owns the status comment; refusing to overwrite it"]

    @pytest.mark.parametrize(
        "artifact_mode",
        [
            "hash-mismatch",
            "mutated-bytes",
            "mutated-report-bytes",
            "metadata-schema-v2",
            "summary-mismatch",
            "extra-file",
            "marker-report",
            "html-report",
            "mention-report",
            "long-report",
            "invalid-utf8-report",
        ],
    )
    def test_publisher_rejects_tampered_or_injectable_artifacts(self, tmp_path: Path, artifact_mode: str):
        state = self._run_publisher(
            tmp_path,
            lint_result="success",
            ppu_result="success",
            new_tree="c" * 40,
            artifact_mode=artifact_mode,
        )
        assert state["failures"] == []
        assert state["gets"] == 1
        assert state["updates"] == 1
        body = state["body"]
        assert "Infrastructure failure: required benchmark artifact is unavailable or invalid" in body
        assert "Running" not in body
        assert body.startswith("<!-- ppu-perf-regression-890p kind=comment request_order=101 run_id=201 run_attempt=1 -->\n")
        assert body.count("<!--") == 1
        assert body.count("ppu-perf-regression-890p kind=") == 1
        for attacker_text in (
            "request_order=999",
            "forged",
            "@everyone",
            "hash mismatch",
            "mutated bytes",
            "post-metadata mutation",
            "legacy metadata schema",
            "forged summary",
            "extra file",
            "A" * 1000,
        ):
            assert attacker_text not in body

    @pytest.mark.parametrize(
        ("artifact_mode", "escaped_fragment"),
        [
            pytest.param("marker-json", "request\\_order=999", id="ownership-marker"),
            pytest.param("html-json", "&lt;details&gt;&lt;summary&gt;forged&lt;/summary&gt;payload&lt;/details&gt;", id="html-fold"),
            pytest.param("mention-json", "&commat;everyone forged mention", id="mention"),
        ],
    )
    def test_publisher_escapes_structured_text_in_fixed_template(self, tmp_path: Path, artifact_mode: str, escaped_fragment: str):
        state = self._run_publisher(
            tmp_path,
            lint_result="success",
            ppu_result="success",
            new_tree="c" * 40,
            artifact_mode=artifact_mode,
        )
        assert state["failures"] == []
        assert state["updates"] == 1
        body = state["body"]
        assert "No regression detected" in body
        assert escaped_fragment in body
        assert body.count("<!--") == 1
        assert "request_order=999" not in body
        assert "<summary>forged</summary>" not in body
        assert "@everyone" not in body
        assert "structured marker fixture" not in body
        assert "structured HTML fixture" not in body
        assert "structured mention fixture" not in body

    def test_publisher_renders_valid_unicode_structured_report_from_fixed_template(self, tmp_path: Path):
        state = self._run_publisher(
            tmp_path,
            lint_result="success",
            ppu_result="success",
            new_tree="c" * 40,
            artifact_mode="valid",
        )
        assert state["failures"] == []
        assert state["gets"] == 1
        assert state["updates"] == 1
        body = state["body"]
        assert "No regression detected" in body
        assert "## Validated Performance Results" in body
        assert "| Old cases (success/total) | 1/1 |" in body
        assert "| New cases (success/total) | 1/1 |" in body
        assert "| Common success | 1 |" in body
        assert "| 基准测试 | examples/ppu/基准测试\\.py | 10.0000 | 8.0000 | 1.2500x | 25.00% | improved |" in body
        assert "**Verdict: PASS**" in body
        assert "仅存在于artifact的Unicode文本" not in body
        assert "## Performance Regression Report" not in body
        assert body.count("<!--") == 1
        assert body.count("ppu-perf-regression-890p kind=") == 1

    def test_publisher_never_forwards_artifact_markdown(self):
        publish = next(
            step for step in self.wf["jobs"]["publish-result"]["steps"] if step["name"] == "Validate artifact and publish result"
        )
        script = publish["with"]["script"]
        assert "const artifactReport = readRegularFile(" in script
        assert "report = renderTrustedReport(compare);" in script
        assert "report = artifactReport" not in script
        assert "escapeMarkdown(row.name" in script
        assert "escapeMarkdown(row.source" in script
        assert "appendList('Performance Regressions'" in script
        assert ".replace(/@/g, '&commat;')" in script

    def test_publisher_only_updates_its_exact_owned_bot_comment(self):
        publish = next(
            step for step in self.wf["jobs"]["publish-result"]["steps"] if step["name"] == "Validate artifact and publish result"
        )
        script = publish["with"]["script"]
        assert "github-actions[bot]" in script
        assert "kind=(comment|push)" in script
        assert "request_order=([1-9][0-9]*)" in script
        assert "process.env.STATUS_COMMENT_ID" in script
        assert "github.rest.issues.getComment" in script
        assert "currentOwnership.requestOrder !== requestOrder" in script
        assert "currentOwnership.runId !== runId" in script
        assert "currentOwnership.runAttempt !== runAttempt" in script
        assert "newer request or run attempt owns the status comment" in script
        assert "updateComment" in script and "createComment" not in script
        assert "listComments" not in script and "paginate" not in script
        assert "body?.includes" not in script and "runMarker" not in script

    def test_old_publisher_cannot_overwrite_new_request_or_attempt(self):
        script = next(
            step["with"]["script"]
            for step in self.wf["jobs"]["publish-result"]["steps"]
            if step["name"] == "Validate artifact and publish result"
        )
        ownership_guard = "currentOwnership.requestOrder !== requestOrder"
        update_call = "await github.rest.issues.updateComment"
        assert ownership_guard in script
        assert script.index(ownership_guard) < script.index(update_call)
        assert "currentOwnership.runId !== runId" in script
        assert "currentOwnership.runAttempt !== runAttempt" in script
        assert "status marker kind mismatch" in script
        assert "Owned status comment has a malformed marker" in script
        assert "Owned status comment is not authored by the trusted bot" in script

    def test_missing_or_invalid_gate_is_fail_closed(self):
        perf_steps = self.wf["jobs"]["ppu-perf"]["steps"]
        verdict = next(step for step in perf_steps if step["name"] == "Gate verdict")
        assert "Gate JSON is missing, invalid, or reports a correctness failure" in verdict["run"]
        publish = next(
            step for step in self.wf["jobs"]["publish-result"]["steps"] if step["name"] == "Validate artifact and publish result"
        )
        script = publish["with"]["script"]
        assert "if (!/^[01]\\n$/.test(rawGateRc))" in script
        assert "gate exit code disagrees with gate.correctness_ok" in script
        assert "Infrastructure failure" in script

    def test_nas_gate_output_validator_rejects_injection_payloads(self, tmp_path: Path):
        fetch = next(step for step in self.wf["jobs"]["ppu-perf"]["steps"] if step["name"] == "Fetch and validate results from NAS")
        command = fetch["run"]
        validator = command.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        marker = tmp_path / "command-substitution-ran"
        payloads = [
            b'";printf${IFS}QA648_INJECTION;#',
            f"$(touch {marker})".encode(),
            b"`printf command-substitution`",
            b"0\nGITHUB_OUTPUT_INJECTED=1\n",
            b"0\r\nGITHUB_OUTPUT_INJECTED=1\n",
            b"9" * 10000,
            b"-1",
            b"",
            b" 0\n",
            b"00",
            b"3\n",
        ]
        for index, payload in enumerate(payloads):
            workspace = tmp_path / f"workspace-{index}"
            source = tmp_path / f"nas-{index}"
            local = workspace / "perf-results"
            source.mkdir()
            local.mkdir(parents=True)
            (source / "gate_rc.txt").write_bytes(payload)
            output = tmp_path / f"github-output-{index}"
            output.touch()
            env = dict(os.environ, GITHUB_WORKSPACE=str(workspace))
            result = subprocess.run(
                [sys.executable, "-c", validator, str(source), str(local), str(output)],
                capture_output=True,
                text=True,
                env=env,
                timeout=10,
            )
            assert result.returncode != 0, payload
            assert output.read_text() == ""
            assert "QA648_INJECTION" not in result.stdout
            assert not marker.exists()

    def test_nas_gate_output_validator_accepts_only_defined_enum(self, tmp_path: Path):
        fetch = next(step for step in self.wf["jobs"]["ppu-perf"]["steps"] if step["name"] == "Fetch and validate results from NAS")
        validator = fetch["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        workspace = tmp_path / "workspace"
        source = tmp_path / "nas"
        local = workspace / "perf-results"
        source.mkdir()
        local.mkdir(parents=True)
        (source / "gate_rc.txt").write_bytes(b"2\n")
        output = tmp_path / "github-output"
        output.touch()
        result = subprocess.run(
            [sys.executable, "-c", validator, str(source), str(local), str(output)],
            capture_output=True,
            text=True,
            env=dict(os.environ, GITHUB_WORKSPACE=str(workspace)),
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert output.read_text().splitlines() == [f"result_dir={local}", "gate_rc=2"]
        assert (local / "gate_rc.txt").read_bytes() == b"2\n"

    def test_shell_blocks_never_interpolate_untrusted_expressions(self):
        executable_blocks = []
        for job in self.wf["jobs"].values():
            for step in job.get("steps", []):
                if "run" in step:
                    executable_blocks.append((step["name"], step["run"]))
                if "command" in step.get("with", {}):
                    executable_blocks.append((step["name"], step["with"]["command"]))
        assert executable_blocks
        for name, block in executable_blocks:
            assert "${{" not in block, f"untrusted expression interpolation in {name}"

    def test_all_output_producers_use_bounded_whitelists(self):
        resolve_script = self.wf["jobs"]["resolve-pr"]["steps"][2]["with"]["script"]
        assert "strictOutput" in resolve_script
        assert "Buffer.byteLength(value, 'utf8') > maxBytes" in resolve_script
        assert "/[\\0\\r\\n]/.test(value)" in resolve_script
        claim_script = self.wf["jobs"]["claim-request"]["steps"][0]["with"]["script"]
        assert "positiveSafeInteger(String(created.id), 'status_comment_id')" in claim_script
        fetch = next(step for step in self.wf["jobs"]["ppu-perf"]["steps"] if step["name"] == "Fetch and validate results from NAS")
        assert 're.fullmatch(r"[012]\\n", raw_gate_rc)' in fetch["run"]
        assert 'output.write(f"gate_rc={gate_rc}\\n")' in fetch["run"]

    def test_publisher_validates_exact_artifact_set_and_every_file_schema(self):
        publish = next(
            step for step in self.wf["jobs"]["publish-result"]["steps"] if step["name"] == "Validate artifact and publish result"
        )
        script = publish["with"]["script"]
        for required in (
            "expectedArtifactFiles",
            "artifact file set does not match the exact publisher schema",
            "O_NOFOLLOW",
            "fstatSync",
            "validatePayload('old_payload.json')",
            "validatePayload('new_payload.json')",
            "payload.schema_version !== 1",
            "payload.cases.length > 100000",
            "artifact SHA-256 digest mismatch",
            "trusted structured report exceeds its bounded size",
            "exactKeys(row",
            "regression_result.png is not a bounded PNG file",
            "invalid canonical gate exit code",
        ):
            assert required in script

    def test_runner_exit_codes_are_captured_not_discarded(self):
        steps = self.wf["jobs"]["ppu-perf"]["steps"]
        scheduler = next(step for step in steps if "ppu-scheduler-action" in step.get("uses", ""))
        command = scheduler["with"]["command"]
        assert "|| OLD_RC=$?" in command
        assert "|| NEW_RC=$?" in command
        assert 'old_payload.json" || true' not in command
        assert 'new_payload.json" || true' not in command

    def test_report_only_policy_has_no_failure_suppression(self):
        workflow_text = (_REPO_ROOT / ".github" / "workflows" / "ppu-perf-regression-bot.yml").read_text()
        assert "continue-on-error" not in workflow_text
        assert "Performance regression detected (Phase 0: report-only)" in workflow_text
        assert "branch-protection" not in workflow_text
        assert "required-check" not in workflow_text

    def test_both_venvs_and_manifest_contract_are_preserved(self):
        steps = self.wf["jobs"]["ppu-perf"]["steps"]
        scheduler = next(step for step in steps if "ppu-scheduler-action" in step.get("uses", ""))
        command = scheduler["with"]["command"]
        assert command.count("python -m venv --system-site-packages") == 2
        assert '"$WORKSPACE/venv_new"' in command
        assert '"$WORKSPACE/venv_old"' in command
        assert 'MANIFEST="$NEW_SRC/maint/scripts/ppu_perf_890p_allowlist.yaml"' in command
        venv_installs = [line.strip() for line in command.splitlines() if line.strip().startswith("pip install pytest ")]
        assert len(venv_installs) == 2
        assert all("pyyaml==6.0.3" in line for line in venv_installs)
        publish = next(
            step for step in self.wf["jobs"]["publish-result"]["steps"] if step["name"] == "Validate artifact and publish result"
        )
        assert "maint/scripts/ppu_perf_890p_allowlist.yaml" in publish["with"]["script"]
        workflow_path = _REPO_ROOT / ".github" / "workflows" / "ppu-perf-regression-bot.yml"
        assert "ppu_perf_890p_allowlist.txt" not in workflow_path.read_text()

    def test_ppu_image_is_torch_210(self):
        expected = "download.t-head.cn/docker_release/pytorch:sdk2.2.0-2.10.0-ubuntu24.04-hggcrt3-py312-20261005"
        assert self.wf.get("env", {}).get("PPU_IMAGE") == expected

    # -------------------------------------------------------------------
    # Push-trigger specific coverage. The push path is restricted by event,
    # repository, and staging branch, and shares no state with issue_comment.
    # -------------------------------------------------------------------

    def test_push_trigger_is_restricted_to_curated_staging_branch(self):
        on = self.wf.get("on") or self.wf.get(True)
        assert on["push"] == {"branches": ["feature/ci-testing-dev"]}
        resolve_if = str(self.wf["jobs"]["resolve-pr"]["if"])
        for required in (
            "github.event_name == 'push'",
            "github.repository == 't-head/tilelang-for-sail'",
            "github.ref == 'refs/heads/feature/ci-testing-dev'",
        ):
            assert required in resolve_if

    def test_push_resolver_revalidates_ref_and_requires_unique_open_pr_and_matching_head_sha(self):
        resolve_text = str(self.wf["jobs"]["resolve-pr"])
        assert "PUSH_ALLOWED_REF = 'refs/heads/feature/ci-testing-dev'" in resolve_text
        assert "strictExactString(context.payload.ref, PUSH_ALLOWED_REF, 'push ref')" in resolve_text
        assert "Push ref gate failed" in resolve_text
        # Query is scoped by state=open and head=owner:branch, then verified.
        assert "github.rest.pulls.list" in resolve_text
        assert "state: 'open'" in resolve_text
        assert "head: `${owner}:${PUSH_ALLOWED_BRANCH}`" in resolve_text
        # Zero / multiple matches must fail closed with a dedicated message.
        assert "requires exactly one open PR" in resolve_text
        # The resolver chains head SHA equality: pr.head.sha == github.sha ==
        # head_commit.id, so a stale PR or a mutated head cannot sneak in.
        assert "pr.head.sha !== context.sha" in resolve_text
        assert "pr.head.sha !== headCommitId" in resolve_text
        assert "headCommitId !== context.sha" in resolve_text

    def test_push_resolver_freezes_dynamic_pr_base(self):
        """Base ref / base SHA always come from the API-reported PR base, so a
        PR targeting dev / main / release is handled identically without any
        hard-coded branch assumption."""
        resolve_text = str(self.wf["jobs"]["resolve-pr"])
        for banned in (
            "'main'",
            '"main"',
            "'dev'",
            '"dev"',
            "'release'",
            '"release"',
        ):
            assert banned not in resolve_text, f"resolver must not hard-code {banned}"
        # It must use the API-reported pr.base.ref / pr.base.sha.
        assert "pr.base.ref" in resolve_text
        assert "pr.base.sha" in resolve_text
        # And must validate the base SHA against the 40-hex-char pattern.
        assert "shaPattern.test(pr.base.sha)" in resolve_text

    def test_push_request_order_is_monotonic_run_id(self):
        resolve_text = str(self.wf["jobs"]["resolve-pr"])
        # Push request_order derives from GITHUB_RUN_ID so a newer run always
        # wins over an older one, mirroring the comment-id monotonicity.
        assert "process.env.GITHUB_RUN_ID" in resolve_text
        assert "requestKind = 'push'" in resolve_text
        assert "`push-${context.sha}`" in resolve_text
        assert "Invalid GITHUB_RUN_ID for push request order" in resolve_text

    def test_push_and_comment_marker_slots_do_not_overwrite_each_other(self):
        claim_script = self.wf["jobs"]["claim-request"]["steps"][0]["with"]["script"]
        publish_script = next(
            step["with"]["script"]
            for step in self.wf["jobs"]["publish-result"]["steps"]
            if step["name"] == "Validate artifact and publish result"
        )
        # The marker format carries ``kind=comment|push``; each slot is only
        # ever read / written for its own kind.
        for script in (claim_script, publish_script):
            assert "kind=(comment|push)" in script
            assert "status marker kind mismatch" in script
            assert "parseOwnership(" in script
        # The claim-request job filters existing status comments by its
        # own kind namespace, so a push's marker is invisible to comment
        # requests and vice versa.
        assert "kindNamespacePattern(requestKind)" in claim_script
        # Status concurrency groups must be scoped by request_kind so the
        # two kinds never serialize into the same slot.
        for job_name in ("claim-request", "publish-result"):
            group = self.wf["jobs"][job_name]["concurrency"]["group"]
            assert "needs.resolve-pr.outputs.request_kind" in group

    def test_push_full_execution_chain_is_identical_to_comment_path(self):
        """A push triggers the same lint + PPU + publisher chain, so the
        trigger does not accidentally short-circuit any execution job."""
        # All execution and status jobs depend on claim-request's should_run,
        # which is the single toggle. No job filters on event_name.
        for name in ("pre-commit-lint", "ppu-perf", "publish-result"):
            if_text = str(self.wf["jobs"][name]["if"])
            assert "needs.claim-request.outputs.should_run == 'true'" in if_text
            assert "github.event_name" not in if_text
        # Publisher receives the same TRIGGER_LABEL/REQUEST_KEY as the
        # claim-request slot (used in the body) so push-produced comments
        # are self-describing on the PR.
        publisher_env = self.wf["jobs"]["publish-result"]["env"]
        for key in ("REQUEST_KIND", "REQUEST_ORDER", "REQUEST_KEY", "TRIGGER_LABEL"):
            assert key in publisher_env

    def test_push_resolver_rejects_unsupported_events_and_wrong_repo(self):
        resolve_text = str(self.wf["jobs"]["resolve-pr"])
        # Repository is also re-validated inside the resolver, not only at
        # the top-level ``if``.
        assert "PUSH_ALLOWED_REPO = 't-head/tilelang-for-sail'" in resolve_text
        assert "Repository mismatch" in resolve_text
        # Unknown events must terminate explicitly.
        assert "Unsupported event" in resolve_text

    def test_push_resolver_metadata_shape_matches_publisher_schema_v3(self):
        resolve_text = str(self.wf["jobs"]["resolve-pr"])
        # Resolver sets an empty comment_id output on push, which the
        # claim-request + publisher scripts translate into metadata.comment_id
        # == null under schema_version 3.
        assert "commentId = ''" in resolve_text
        # Publisher's metadata validation expects schema_version === 3 and
        # a null comment_id for push kind.
        publisher = next(
            step for step in self.wf["jobs"]["publish-result"]["steps"] if step["name"] == "Validate artifact and publish result"
        )
        script = publisher["with"]["script"]
        assert "'schema_version'" in script
        assert "'request_kind'" in script
        assert "'request_order'" in script
        assert "expectedCommentField = requestKind === 'comment' ? commentId : null" in script


# ===========================================================================
# Tests for the tracked repository ci.yml (functional CI)
# ===========================================================================


class TestFunctionalCIYAML:
    """Structural assertions on the tracked functional CI workflow."""

    @pytest.fixture(autouse=True)
    def load_ci(self):
        import yaml

        wf_path = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
        with open(wf_path) as f:
            self.ci = yaml.safe_load(f)

    def test_ppu_tests_job_exists(self):
        """ppu-tests job exists in the functional CI."""
        jobs = self.ci["jobs"]
        assert "ppu-tests" in jobs
        assert "lint" in jobs
        # On some branches ppu-tests has needs:[lint]; on dev baseline it may not.
        needs = jobs["ppu-tests"].get("needs")
        if needs is not None:
            if isinstance(needs, str):
                needs = [needs]
            assert "lint" in needs

    def test_ppu_tests_has_no_bypass_conditions(self):
        """ppu-tests must not bypass needs failure propagation."""
        job = self.ci["jobs"]["ppu-tests"]
        # No job-level if that would unlock the job when lint fails/cancels
        job_if = job.get("if", "")
        assert "always()" not in str(job_if)
        assert "!cancelled()" not in str(job_if)
        assert "cancelled() == false" not in str(job_if)
        assert "continue-on-error" not in job

    def test_ppu_tests_matrix_defaults_two_boards(self):
        """The matrix fallback still provides both ZW-M890P and OAM-810E defaults."""
        job = self.ci["jobs"]["ppu-tests"]
        board_expr = job["strategy"]["matrix"]["board"]
        assert "ZW-M890P" in board_expr
        assert "OAM-810E" in board_expr
        assert "inputs.board_matrix" in board_expr

    def test_scheduler_step_only_in_ppu_tests(self):
        """ppu-scheduler-action is still only reachable through ppu-tests."""
        scheduler_jobs = []
        for job_name, job in self.ci["jobs"].items():
            steps = job.get("steps", []) or []
            if any("ppu-scheduler-action" in step.get("uses", "") for step in steps):
                scheduler_jobs.append(job_name)
        assert scheduler_jobs == ["ppu-tests"]

    def test_lint_job_contract_preserved(self):
        """The baseline lint job still carries the exact pre-commit contract."""
        lint = self.ci["jobs"]["lint"]
        assert lint["runs-on"] == "ubuntu-latest"
        assert lint["timeout-minutes"] == 30
        runs = [step.get("run", "") for step in lint["steps"] if "run" in step]
        assert any("-m compileall -q -f tilelang" in r for r in runs)
        assert any("pipx run pre-commit run --all-files --color=always --show-diff-on-failure" in r for r in runs)

    def test_on_block_matches_upstream_triggers(self):
        """Functional CI runs for PR updates or manual dispatch, never push."""
        on = self.ci.get("on") or self.ci.get(True)
        assert on is not None
        assert set(on) == {"pull_request", "workflow_dispatch"}
        assert "push" not in on
        assert on["pull_request"] == {
            "types": [
                "labeled",
                "unlabeled",
                "opened",
                "synchronize",
                "reopened",
                "ready_for_review",
            ]
        }


# ===========================================================================
# Tests for the shipped PPU 890P allowlist manifest
# ===========================================================================

_MANIFEST_PATH = _SCRIPTS_DIR / "ppu_perf_890p_allowlist.yaml"

_EXPECTED_PPU_ENTRIES = (
    "examples/ppu/flash_decoding/regression_example_flash_decoding.py",
    "examples/ppu/seer_attention/regression_block_sparse_attn_tilelang.py",
)

_EXPECTED_NEW_ENTRIES = (
    "examples/attention_sink/regression_attention_sink.py",
    "examples/flash_attention/regression_example_flash_attention.py",
    "examples/flash_decoding/regression_example_flash_decoding.py",
    "examples/seer_attention/regression_block_sparse_attn_tilelang.py",
    "examples/gemm_splitk/regression_example_gemm_splitk.py",
    "examples/blocksparse_gemm/regression_example_blocksparse_gemm.py",
    "examples/fusedmoe/regression_example_fusedmoe.py",
    "examples/deepseek_mhc/regression_example_mhc.py",
)

_EXPECTED_EXCLUDED_ENTRIES = (
    "examples/linear_attention/regression_linear_attn.py",
    "examples/minference/regression_vs_sparse_attn.py",
    "examples/gemm/regression_example_gemm.py",
    "examples/dequantize_gemm/regression_example_dequantize_gemm.py",
    "examples/deepseek_mla/regression_example_mla_decode.py",
    "examples/deepseek_v32/regression_tilelang_example_deepseek_v32.py",
    "examples/ppu/topk/regression_topk_tilelang.py",
    "examples/blocksparse_attention/regression_example_blocksparse_attention.py",
    "examples/gemm_fp8/regression_example_gemm_fp8.py",
    "examples/deepseek_nsa/regression_example_tilelang_nsa.py",
)


class TestShippedManifest:
    """Structural assertions on the tracked PPU 890P YAML manifest."""

    @pytest.fixture(autouse=True)
    def load_manifest(self):
        self.document = yaml.safe_load(_MANIFEST_PATH.read_text())
        self.resolved = rcr.load_manifest(str(_MANIFEST_PATH), _REPO_ROOT)
        self.relative = [str(path.resolve().relative_to(_REPO_ROOT.resolve())) for path in self.resolved]

    def test_schema_header(self):
        assert set(self.document) == {
            "schema_version",
            "platform",
            "benchmarks",
            "excluded",
        }
        assert self.document["schema_version"] == 1
        assert self.document["platform"] == "ppu-890p"

    def test_exact_enabled_and_excluded_counts(self):
        assert len(self.document["benchmarks"]) == 10
        assert len(self.document["excluded"]) == 10
        assert len(self.resolved) == 10

    def test_enabled_entry_schema(self):
        for entry in self.document["benchmarks"]:
            assert set(entry) <= {"path", "group", "expected_cases"}
            assert isinstance(entry.get("path"), str) and entry["path"].strip()
            if "group" in entry:
                assert isinstance(entry["group"], str) and entry["group"].strip()
            if "expected_cases" in entry:
                assert type(entry["expected_cases"]) is int
                assert entry["expected_cases"] > 0

    def test_excluded_entry_schema_and_reasons(self):
        for entry in self.document["excluded"]:
            assert set(entry) == {"path", "reason"}
            assert isinstance(entry["path"], str) and entry["path"].strip()
            assert isinstance(entry["reason"], str) and entry["reason"].strip()

    def test_no_duplicate_or_overlapping_paths(self):
        enabled = [entry["path"] for entry in self.document["benchmarks"]]
        excluded = [entry["path"] for entry in self.document["excluded"]]
        assert len(enabled + excluded) == len(set(enabled + excluded))

    def test_enabled_entries_match_verified_set(self):
        expected = set(_EXPECTED_PPU_ENTRIES) | set(_EXPECTED_NEW_ENTRIES)
        assert set(self.relative) == expected

    def test_excluded_entries_match_expected_set(self):
        excluded = {entry["path"] for entry in self.document["excluded"]}
        assert excluded == set(_EXPECTED_EXCLUDED_ENTRIES)
        assert excluded.isdisjoint(self.relative)

    def test_enabled_entries_are_existing_regression_files(self):
        for path in self.resolved:
            assert path.is_file(), f"Not a file: {path}"
            assert path.name.startswith("regression_") and path.name.endswith(".py")

    def test_old_txt_manifest_is_removed(self):
        assert not (_SCRIPTS_DIR / "ppu_perf_890p_allowlist.txt").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
