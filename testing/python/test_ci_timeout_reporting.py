from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    source = "from __future__ import annotations\n" + path.read_text(encoding="utf-8")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


TEST_RUNNER = _load_script("_tilelang_ci_test_runner", ".github/scripts/test_tilelang.py")
REPORTER = _load_script("_tilelang_ci_reporter", ".github/scripts/report_test_results.py")


def _write_pass_xml(path: Path) -> None:
    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root,
        "testsuite",
        name="pytest",
        tests="1",
        failures="0",
        errors="0",
        skipped="0",
        time="0.100",
    )
    ET.SubElement(
        suite,
        "testcase",
        classname="testing.python.test_ok",
        file="testing/python/test_ok.py",
        name="test_ok",
        time="0.100",
    )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def test_write_timeout_xml_has_failure_metadata_and_escaping(tmp_path: Path):
    xml_path = tmp_path / "timeout.xml"
    config = TEST_RUNNER.TestConfig(
        file_path="examples/a&b/test_<kernel>.py",
        test_filter="TestSuite::test_case[param<&>]",
    )

    returned = TEST_RUNNER._write_timeout_xml(config, 17, 17.25, str(xml_path))

    assert returned == str(xml_path)
    raw_xml = xml_path.read_text(encoding="utf-8")
    assert "a&amp;b" in raw_xml
    assert "test_&lt;kernel&gt;.py" in raw_xml

    root = ET.parse(xml_path).getroot()
    suite = root.find("testsuite")
    assert suite is not None
    assert suite.attrib == {
        "name": "pytest-timeout",
        "tests": "1",
        "failures": "1",
        "errors": "0",
        "skipped": "0",
        "time": "17.250",
    }
    testcase = suite.find("testcase")
    assert testcase is not None
    assert testcase.get("classname") == config.file_path
    assert testcase.get("file") == config.file_path
    assert testcase.get("name") == config.test_filter
    assert testcase.get("time") == "17.250"

    failure = testcase.find("failure")
    assert failure is not None
    assert failure.get("type") == "TimeoutError"
    assert failure.get("message") == "Test timed out after 17 seconds"
    assert "forcibly terminated with SIGKILL" in (failure.text or "")
    assert TEST_RUNNER.parse_junit_xml_counts(str(xml_path)) == (1, 1, 0, 0)

    no_filter_xml = tmp_path / "timeout_no_filter.xml"
    TEST_RUNNER._write_timeout_xml(TEST_RUNNER.TestConfig("testing/python/test_file.py"), 3, 3.0, str(no_filter_xml))
    no_filter_cases = ET.parse(no_filter_xml).findall(".//testcase")
    assert len(no_filter_cases) == 1
    assert no_filter_cases[0].get("name") == "timeout"


def test_merge_and_report_include_timeout_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pass_xml = tmp_path / "pass.xml"
    timeout_xml = tmp_path / "timeout.xml"
    merged_xml = tmp_path / "merged.xml"
    _write_pass_xml(pass_xml)
    config = TEST_RUNNER.TestConfig(
        file_path="examples/hanging/test_timeout.py",
        test_filter="test_timeout",
    )
    TEST_RUNNER._write_timeout_xml(config, 600, 600.5, str(timeout_xml))

    assert TEST_RUNNER.merge_junit_xml([str(pass_xml), str(timeout_xml)], str(merged_xml))
    merged_root = ET.parse(merged_xml).getroot()
    merged_suite = merged_root.find("testsuite")
    assert merged_suite is not None
    assert merged_suite.get("tests") == "2"
    assert merged_suite.get("failures") == "1"
    timeout_case = next(tc for tc in merged_root.findall(".//testcase") if tc.find("failure") is not None)
    assert timeout_case.get("classname") == config.file_path
    assert timeout_case.get("file") == config.file_path
    assert timeout_case.find("failure").get("type") == "TimeoutError"

    parsed_cases, stats = REPORTER._parse_xml(str(merged_xml))
    parsed_timeout = next(tc for tc in parsed_cases if tc["status"] == "fail")
    assert stats == {"total": 2, "passed": 1, "failed": 1, "skipped": 0, "time": 600.6}
    assert parsed_timeout["name"] == "examples_hanging_test_timeout_py_test_timeout"
    assert "Test timed out after 600 seconds" in parsed_timeout["message"]
    assert "TimeoutError" in parsed_timeout["message"]

    args = SimpleNamespace(
        title="Timeout report",
        run_number="",
        branch="",
        env_info="",
        docker_image="",
        xml=str(merged_xml),
        job_status="",
        duration="",
        pods="",
    )
    assert REPORTER._report_single_board(args) == 1
    output = capsys.readouterr().out
    assert "examples_hanging_test_timeout_py_test_timeout" in output
    assert "Test timed out after 600 seconds" in output
    assert "TimeoutError" in output


@pytest.mark.parametrize(
    "residual",
    [
        "<?xml version='1.0'?><testsuites><testsuite tests='1' failures='0' errors='0' skipped='0'><testcase classname='old' name='same_case'/></testsuite></testsuites>",
        "<?xml version='1.0'?><testsuites><testsuite>",
    ],
    ids=["parseable", "truncated"],
)
def test_timeout_xml_replaces_residual_without_duplicates(tmp_path: Path, residual: str):
    xml_path = tmp_path / "results_0.xml"
    xml_path.write_text(residual, encoding="utf-8")
    config = TEST_RUNNER.TestConfig("testing/python/test_hang.py", "same_case")

    TEST_RUNNER._write_timeout_xml(config, 8, 8.1, str(xml_path))

    root = ET.parse(xml_path).getroot()
    testcases = root.findall(".//testcase")
    assert len(testcases) == 1
    assert testcases[0].get("classname") == config.file_path
    assert testcases[0].get("name") == config.test_filter
    assert testcases[0].find("failure").get("type") == "TimeoutError"


def test_run_python_test_timeout_keeps_failure_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class TimeoutProcess:
        pid = 1234
        returncode = None

        def communicate(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="pytest", timeout=timeout)
            self.returncode = -9
            return "partial output", ""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(TEST_RUNNER.subprocess, "Popen", lambda *args, **kwargs: TimeoutProcess())
    monkeypatch.setattr(TEST_RUNNER.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(TEST_RUNNER.os, "killpg", lambda pgid, sig: None)
    config = TEST_RUNNER.TestConfig("testing/python/test_hang.py", "test_hang")

    result = TEST_RUNNER._run_python_test(config, 0, timeout=2)

    assert result.timed_out is True
    assert result.returncode == -9
    assert result.failed is True
    assert result.xml_path == "results_0.xml"
    assert (result.xml_tests, result.xml_failures, result.xml_errors, result.xml_skipped) == (1, 1, 0, 0)
    failure = ET.parse(result.xml_path).find(".//failure")
    assert failure is not None
    assert failure.get("message") == "Test timed out after 2 seconds"

    fail_list = tmp_path / "fail_list.json"
    TEST_RUNNER.dump_fail_list([result], str(fail_list))
    assert json.loads(fail_list.read_text(encoding="utf-8")) == [
        {
            "file_path": config.file_path,
            "test_filter": config.test_filter,
            "extra_args": [],
            "reason": "TIMEOUT",
        }
    ]
