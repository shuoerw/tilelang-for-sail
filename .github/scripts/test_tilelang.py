#!/usr/bin/env python3

import argparse
import concurrent.futures
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field


# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------

# 单个用例最长运行时间（秒），超时后强制终止并判定为 FAIL
TEST_TIMEOUT_SECONDS = 1 * 60 * 10


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class TestConfig:
    """单个测试的配置信息"""

    file_path: str  # 测试文件路径
    test_filter: str | None = None  # pytest -k 过滤或 ::node_id
    extra_args: list[str] = field(default_factory=list)  # 额外 pytest 参数

    @property
    def display_name(self) -> str:
        """用于日志输出的可读名称"""
        name = self.file_path
        if self.test_filter:
            name = f"{name}::{self.test_filter}"
        return name


@dataclass
class TestResult:
    """单个测试运行的结果"""

    config: TestConfig
    returncode: int = -1
    duration: float = 0.0
    xml_path: str | None = None
    stdout: str = ""
    stderr: str = ""
    # XML-parsed counts for accurate pass/fail/skip determination
    xml_tests: int = 0
    xml_failures: int = 0
    xml_errors: int = 0
    xml_skipped: int = 0
    timed_out: bool = False  # 用例超时被强制终止

    @property
    def passed(self) -> bool:
        return self.returncode == 0

    @property
    def skipped(self) -> bool:
        """Test is considered skipped if all tests were skipped (none failed/errored)."""
        # 超时的用例一律判定为 FAIL，不算跳过
        if self.timed_out:
            return False
        # pytest exit code 5 means no tests were collected (all deselected/skipped)
        if self.returncode == 5:
            return True
        # If XML was parsed and shows all tests are skipped with no real failures
        if (
            self.xml_tests > 0
            and self.xml_failures == 0
            and self.xml_errors == 0
            and self.xml_skipped > 0
            and self.xml_skipped == self.xml_tests
        ):
            return True
        # Non-zero return code but XML shows no failures/errors, only skips
        return self.returncode != 0 and self.xml_tests > 0 and self.xml_failures == 0 and self.xml_errors == 0 and self.xml_skipped > 0

    @property
    def failed(self) -> bool:
        """Test is a real failure (not passed, not skipped)."""
        return not self.passed and not self.skipped

    @property
    def status_label(self) -> str:
        # 1. 显式失败 / 错误 / 超时 → FAIL
        if self.xml_failures > 0 or self.xml_errors > 0:
            return "FAIL"
        if self.timed_out:
            return "FAIL"
        # 2. 任意 skip（含"部分 skip"和"全 skip"）优先于 PASS，
        #    避免 returncode == 0 但存在被 pytest 跳过的用例时被误判成 PASS
        if self.xml_skipped > 0:
            return "SKIP"
        if self.returncode == 5:
            return "SKIP"
        # 3. 兜底判定
        if self.returncode == 0:
            return "PASS"
        return "FAIL"


# ---------------------------------------------------------------------------
# scan_cases
# ---------------------------------------------------------------------------

test_dir = os.getcwd()

# PPU-specific: exclude paths that cause collection errors due to
# missing dependencies (fla) or vendored TVM incompatibilities.
# NOTE: entries here are hidden from pytest entirely (no collection attempt,
# no failure record). Anything that MUST be surfaced as a CI failure via the
# collection-error interception path below MUST NOT be listed here.
COLLECT_IGNORE_PATHS = [
    "examples/ppu/linear_attention",
    "testing/python/transform/test_tilelang_transform_reorder_aiu_loads.py",
]


@dataclass
class CollectionError:
    """A single pytest collection failure captured from --collect-only output.

    file_path is normalized to the same layout as TestConfig.file_path so it
    dedupes cleanly across per-directory scans. traceback holds the full
    diagnostic text with ANSI escape sequences already stripped (via
    _strip_ansi) so JUnit XML, fail_list JSON, and log output are clean.
    XML escaping happens at write time via ElementTree.
    """

    file_path: str
    traceback: str


# ---------------------------------------------------------------------------
# ANSI escape sequence stripping
# ---------------------------------------------------------------------------

# Matches CSI (Control Sequence Introducer) sequences including SGR (Select
# Graphic Rendition) — e.g. \x1b[1m, \x1b[31;1m, \x1b[0m — as well as
# OSC (Operating System Command) sequences and other 7-bit C1 escapes.
_ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _strip_ansi(text: str) -> str:
    """Remove all ANSI CSI/SGR control sequences from *text*.

    Used to sanitize pytest output captured under FORCE_COLOR / --color=yes
    before regex matching and before storing text in JUnit XML / fail_list.
    """
    return _ANSI_RE.sub("", text)


# Regex matching the pytest `_____ ERROR collecting <path> _____` block header.
# pytest uses at least one leading underscore on each side; be tolerant of
# trailing whitespace and any amount of underscore padding.
_COLLECT_ERROR_HEADER_RE = re.compile(r"^_+\s*ERROR collecting\s+(.+?)\s*_+\s*$")
# Boundaries that terminate a captured traceback block.
_COLLECT_ERROR_TERMINATOR_RE = re.compile(
    r"^(=+\s*(short test summary info|ERRORS|FAILURES|warnings summary|passed|failed)\b.*=+\s*$"
    r"|!!!+\s*Interrupted.*!!!+\s*$"
    r"|=+\s*\d+ .* in [\d.]+s\s*=+\s*$)"
)
# Short-summary `ERROR <path>[::...]` lines emitted in the trailing summary.
_COLLECT_ERROR_SUMMARY_RE = re.compile(r"^ERROR\s+(\S+\.py)(?:\s*-.*|\s*::.*|\s*)$")


def _parse_collection_output(output: str):
    """Parse pytest --collect-only -q output.

    Returns:
        test_lines: List[str] of raw `file::node_id` lines that succeeded.
        error_files: Dict[str, str] mapping pytest-reported file path
                     (relative to pytest rootdir) to its traceback text.

    Deduplication: the same file surfaced in both the `___ ERROR collecting X ___`
    block header and the short-summary `ERROR X` line resolves to a single entry
    with the traceback text preferred over the summary marker.
    """
    test_lines: list[str] = []
    error_files: dict = {}

    # Strip ANSI from the entire output once so every subsequent regex match
    # and stored text is free of escape sequences (CI sets FORCE_COLOR=1).
    clean_output = _strip_ansi(output)

    lines = clean_output.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        stripped = raw.strip()

        m = _COLLECT_ERROR_HEADER_RE.match(stripped)
        if m:
            err_path = m.group(1).strip()
            tb_lines: list[str] = []
            i += 1
            while i < n:
                nxt = lines[i]
                nxt_stripped = nxt.strip()
                if _COLLECT_ERROR_HEADER_RE.match(nxt_stripped):
                    break
                if _COLLECT_ERROR_TERMINATOR_RE.match(nxt_stripped):
                    break
                tb_lines.append(nxt)
                i += 1
            traceback_text = "\n".join(tb_lines).strip()
            # Prefer the block-header traceback over any prior short-summary stub
            if err_path not in error_files or not error_files[err_path].strip():
                error_files[err_path] = traceback_text or f"ERROR collecting {err_path} (empty traceback)"
            continue

        m2 = _COLLECT_ERROR_SUMMARY_RE.match(stripped)
        if m2:
            err_path = m2.group(1).strip()
            if err_path not in error_files:
                error_files[err_path] = f"pytest short summary: {stripped}"
            i += 1
            continue

        if "::" in stripped and stripped:
            test_lines.append(stripped)
        i += 1

    return test_lines, error_files


def collect_tests(target_dir):
    """Run pytest --collect-only and return (tests, collection_errors).

    Contract:
      * Successfully collected node ids become TestConfig entries (unchanged).
      * Every `ERROR collecting <path>` block becomes a CollectionError so the
        caller can surface it as a real failure in JUnit / fail_list.
      * If pytest exits non-zero and we cannot parse any error entries, we
        synthesize a directory-level CollectionError to fail closed (no silent
        loss of failure signal).
    """
    cmd = ["pytest", target_dir[0], "--collect-only", "-q"]
    for ignore_path in COLLECT_IGNORE_PATHS:
        cmd.extend(["--ignore", ignore_path])
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    test_lines, error_files = _parse_collection_output(result.stdout or "")

    tests: list[TestConfig] = []
    for line in test_lines:
        if "::" not in line:
            continue
        file_part, test_part = line.split("::", 1)
        tests.append(
            TestConfig(
                file_path=os.path.normpath(os.path.join(target_dir[1], file_part)),
                test_filter=test_part,
            )
        )

    collection_errors: list[CollectionError] = []
    for err_path, tb in error_files.items():
        norm_path = os.path.normpath(os.path.join(target_dir[1], err_path))
        collection_errors.append(CollectionError(file_path=norm_path, traceback=tb))

    # ---- Handle non-zero return codes ------------------------------------
    # Strip ANSI once for all diagnostic / log output below.
    clean_stdout = _strip_ansi(result.stdout or "")

    if result.returncode != 0:
        if not collection_errors and not tests:
            if result.returncode == 5:
                # Scenario A-1: exit code 5 = "no tests were collected" with
                # NO real ERROR-collecting blocks.  This is benign — the
                # directory simply has no runnable tests (e.g. all filtered
                # by board skip / markers / conftest, or no test files
                # present).  Do NOT fabricate a CollectionError, and do NOT
                # emit any result-shaped line: this directory produces no
                # TestConfig and no CollectionError, so it stays invisible in
                # print_summary / fail_list / skip_list.  Only surface a
                # debug breadcrumb when TILELANG_TEST_DEBUG is explicitly set,
                # so the default run keeps the final result display clean.
                if os.environ.get("TILELANG_TEST_DEBUG"):
                    print(
                        f"DEBUG [collect_tests] target={target_dir[0]} | "
                        f"returncode=5 | parsed_tests=0 | parsed_errors=0 | "
                        f"action=no_tests_after_filter | "
                        f"note=no ERROR collecting blocks found, treating as all-filtered"
                    )
            else:
                # Scenario A-2: true zero-collection with a non-5 failure
                # code — something unexpected went wrong.  Synthesize a
                # directory-level CollectionError so it enters JUnit /
                # fail_list and flips CI to failure.
                synth_path = os.path.normpath(target_dir[0])
                collection_errors.append(
                    CollectionError(
                        file_path=synth_path,
                        traceback=(
                            f"pytest --collect-only exited with returncode={result.returncode} "
                            f"for {target_dir[0]} but no tests and no ERROR collecting "
                            f"entries could be parsed from the output.\n\n"
                            f"Full pytest output follows:\n{clean_stdout}"
                        ),
                    )
                )
                print(
                    f"ERROR [collect_tests] target={target_dir[0]} | "
                    f"returncode={result.returncode} | "
                    f"parsed_tests=0 | parsed_errors=0 | "
                    f"action=synthetic_collection_error\n"
                    f"--- begin diagnostic output ---\n"
                    f"{clean_stdout}\n"
                    f"--- end diagnostic output ---"
                )
        elif not collection_errors and tests:
            # Scenario B: rc non-zero (e.g. rc=5 "no tests ran") but the
            # parser extracted valid node ids.  Don't fabricate a collection
            # error — warn and proceed with the collected tests.
            print(
                f"WARNING [collect_tests] target={target_dir[0]} | "
                f"returncode={result.returncode} | "
                f"parsed_tests={len(tests)} | parsed_errors=0 | "
                f"action=continue_with_parsed_tests | "
                f"note=return code inconsistent with parsed test count"
            )
        else:
            # Scenario C/D: real collection errors parsed (possibly alongside
            # valid tests).  Keep them; caller surfaces them as CI failures.
            # Valid tests, if any, will still be executed.
            print(
                f"WARNING [collect_tests] target={target_dir[0]} | "
                f"returncode={result.returncode} | "
                f"parsed_tests={len(tests)} | "
                f"parsed_errors={len(collection_errors)} | "
                f"action=keep_real_errors_and_continue"
            )

    return tests, collection_errors


def load_skip():
    platform = os.environ.get("CI_TEST_PLATFORM")
    skip_list_file = f"{platform}_skip.json"
    if not os.path.exists(skip_list_file):
        print(f"skip list file not exists: {skip_list_file}")
        return set()

    with open(skip_list_file) as f:
        data = json.load(f)

    return {(x["file_path"], x["test_filter"]) for x in data}


def scan_cases(test_dir: tuple[str, str]) -> tuple[list[TestConfig], list[CollectionError]]:
    collected, errors = collect_tests(test_dir)
    print(f"Collected test: {len(collected)}, collection errors: {len(errors)}")

    skip_set = load_skip()
    print(f"skipped test: {len(skip_set)}")

    filtered = [t for t in collected if (t.file_path, t.test_filter) not in skip_set]
    print(f"filtered test: {len(filtered)}")

    return filtered, errors


# ---------------------------------------------------------------------------
# XML 结果解析
# ---------------------------------------------------------------------------


def parse_junit_xml_counts(xml_path: str) -> tuple[int, int, int, int]:
    """
    解析 JUnit XML 文件，提取测试统计数据。

    通过检查每个 <testcase> 的子元素来确定状态：
      - <failure> 子元素 → 失败
      - <error> 子元素 → 错误
      - <skipped> 子元素 → 跳过
      - 无上述子元素 → 通过

    参数:
        xml_path: JUnit XML 文件路径

    返回:
        (tests, failures, errors, skipped) 元组
    """
    if not xml_path or not os.path.exists(xml_path):
        return (0, 0, 0, 0)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except (ET.ParseError, Exception):
        return (0, 0, 0, 0)

    # Collect all <testsuite> elements
    testsuites: list[ET.Element] = []
    if root.tag == "testsuites":
        testsuites = root.findall("testsuite")
    elif root.tag == "testsuite":
        testsuites = [root]
    else:
        return (0, 0, 0, 0)

    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0

    for ts in testsuites:
        for tc in ts.findall("testcase"):
            total_tests += 1
            # Check child elements to determine status
            if tc.find("failure") is not None:
                total_failures += 1
            elif tc.find("error") is not None:
                total_errors += 1
            elif tc.find("skipped") is not None:
                total_skipped += 1
            # else: passed (no child element indicating failure/error/skip)

    return (total_tests, total_failures, total_errors, total_skipped)


def _write_timeout_xml(
    config: TestConfig,
    timeout: int | float,
    duration: float,
    xml_path: str,
) -> str:
    """Write a canonical single-testcase JUnit failure for a timed-out run.

    A pytest process normally writes JUnit XML only during session shutdown, so
    SIGKILL often leaves no file or a truncated one. Each process here executes
    exactly one TestConfig; replacing any residual XML avoids duplicating that
    case while also guaranteeing malformed partial XML cannot break the merge.
    """
    timeout_label = f"{timeout:g}"
    elapsed = max(duration, 0.0)
    message = f"Test timed out after {timeout_label} seconds"

    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root,
        "testsuite",
        name="pytest-timeout",
        tests="1",
        failures="1",
        errors="0",
        skipped="0",
        time=f"{elapsed:.3f}",
    )
    testcase = ET.SubElement(
        suite,
        "testcase",
        classname=config.file_path,
        file=config.file_path,
        name=config.test_filter or "timeout",
        time=f"{elapsed:.3f}",
    )
    failure = ET.SubElement(
        testcase,
        "failure",
        type="TimeoutError",
        message=message,
    )
    failure.text = (
        f"TimeoutError: The pytest process for {config.display_name} exceeded "
        f"{timeout_label} seconds and was forcibly terminated with SIGKILL."
    )

    tree = ET.ElementTree(root)
    with contextlib.suppress(AttributeError):
        ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path


# ---------------------------------------------------------------------------
# 测试执行
# ---------------------------------------------------------------------------


def run_single_test(
    config: TestConfig,
    index: int,
    verbose: bool = False,
    timeout: int = TEST_TIMEOUT_SECONDS,
) -> TestResult:
    """
    运行单个测试（Python pytest 或二进制可执行文件），生成临时 JUnit XML 结果文件。

    根据 config.test_type 分派到对应的执行逻辑：
      - "python": 使用 pytest --junitxml 生成 XML
      - "binary": 运行二进制，尝试 gtest XML 输出或自动生成 XML

    参数:
        config:  测试配置
        index:   测试序号（用于生成唯一临时文件名）
        verbose: 是否输出详细信息
        timeout: 单个用例的最长运行时间（秒），超时后强制终止并判定为 FAIL

    返回:
        TestResult 包含执行结果、计时和 XML 路径
    """
    return _run_python_test(config, index, verbose, timeout)


def _run_python_test(
    config: TestConfig,
    index: int,
    verbose: bool = False,
    timeout: int = TEST_TIMEOUT_SECONDS,
) -> TestResult:
    """
    运行单个 pytest 测试，生成临时 JUnit XML 结果文件。

    若运行时间超过 timeout 秒，则强制终止进程并判定为 FAIL。
    """
    result = TestResult(config=config)
    temp_xml = f"results_{index}.xml"

    # 构建 pytest 命令
    target = config.file_path
    if config.test_filter:
        target = f"{target}::{config.test_filter}"

    cmd: list[str] = [
        "pytest",
        "-v",
        "--color=yes",
        "--durations=0",
        "--showlocals",
        target,
        f"--junitxml={temp_xml}",
    ]
    # 追加额外参数（如 -x, --timeout 等）
    if config.extra_args:
        cmd.extend(config.extra_args)

    # 打印运行信息
    print(f"\n{'=' * 70}")
    print(f"[{index + 1}] 正在运行 [Python]: {config.display_name}")
    print(f"    命令: {' '.join(cmd)}")
    if verbose:
        print(f"    文件路径: {config.file_path} | 存在: {os.path.exists(config.file_path)}")
    print(f"{'=' * 70}")

    start_time = time.time()
    proc = None
    try:
        # 使用 Popen + 进程组，确保超时时能杀掉整个进程树（含 GPU 子进程）
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,  # 创建新进程组
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        result.returncode = proc.returncode
        result.stdout = stdout or ""
        result.stderr = stderr or ""

        # verbose 模式：正常结束后打印完整输出
        if verbose and stdout:
            print(stdout)

        if proc.returncode == 0:
            print(f"  ✅ 测试通过: {config.display_name}")
        elif proc.returncode == 5:
            print(f"  ⏭️ 无测试被收集 (返回码=5): {config.display_name}")
        else:
            print(f"  ❌ 测试失败 (返回码={proc.returncode}): {config.display_name}")
            # 失败时打印 stderr 帮助调试
            if result.stderr:
                print("  --- stderr 输出 (最后 20 行) ---")
                for line in result.stderr.strip().splitlines()[-20:]:
                    print(f"    {line}")

    except subprocess.TimeoutExpired:
        # 超时：杀掉整个进程组（包含孙进程）
        if proc is not None:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            # 收集超时前已有的输出
            stdout, stderr = proc.communicate()
            result.stdout = stdout or ""
            result.stderr = stderr or ""
        result.timed_out = True
        result.returncode = -9  # 以负返回码标记被强制终止
        print(f"  ❌ 测试超时 (超过 {timeout} 秒)，已强制终止并判定为 FAIL: {config.display_name}")
        # 打印超时前捕获的部分输出
        if result.stdout:
            print("  --- stdout 输出 (最后 20 行) ---")
            for line in result.stdout.strip().splitlines()[-20:]:
                print(f"    {line}")
    except FileNotFoundError:
        print("  ❌ 找不到 pytest 命令，请确认已安装 pytest")
        result.returncode = -1
    except Exception as e:
        print(f"  ❌ 执行异常: {e}")
        result.returncode = -1
    finally:
        # 无论成功或失败，都记录生成的 XML 文件
        result.duration = time.time() - start_time
        if result.timed_out:
            _write_timeout_xml(config, timeout, result.duration, temp_xml)
        if os.path.exists(temp_xml):
            result.xml_path = temp_xml
            # Parse XML to get accurate test counts (pass/fail/skip)
            counts = parse_junit_xml_counts(temp_xml)
            result.xml_tests, result.xml_failures, result.xml_errors, result.xml_skipped = counts
            if verbose:
                print(f"  📄 已生成临时 XML: {temp_xml}")
                print(f"     XML 统计: tests={counts[0]}, failures={counts[1]}, errors={counts[2]}, skipped={counts[3]}")
        else:
            if verbose:
                print(f"  ⚠️ 未生成 XML 文件: {temp_xml}")

    # Re-evaluate status after XML parsing: skipped tests aren't failures
    if result.skipped and not result.passed:
        print(f"  ⏭️ 测试被跳过 (非失败): {config.display_name}")

    return result


# ---------------------------------------------------------------------------
# JUnit XML 合并
# ---------------------------------------------------------------------------


def merge_junit_xml(
    xml_files: list[str],
    output_path: str,
    verbose: bool = False,
) -> bool:
    """
    将多个 JUnit XML 结果文件合并为一个。

    支持两种常见 pytest 输出结构:
      - <testsuites><testsuite>...</testsuite></testsuites>
      - <testsuite>...</testsuite>  (直接作为根元素)

    参数:
        xml_files:   待合并的 XML 文件路径列表
        output_path: 合并后的输出文件路径
        verbose:     是否输出详细信息

    返回:
        合并是否成功（True/False）
    """
    # 初始化聚合统计
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    total_time = 0.0
    all_testcases: list[ET.Element] = []

    if not xml_files:
        print("⚠️ 没有 XML 文件需要合并，将生成空的结果文件")
    else:
        for xml_file in xml_files:
            # 检查文件是否存在且非空
            if not os.path.exists(xml_file):
                print(f"  ⚠️ 跳过不存在的文件: {xml_file}")
                continue
            if os.path.getsize(xml_file) == 0:
                print(f"  ⚠️ 跳过空文件: {xml_file}")
                continue

            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
            except ET.ParseError as e:
                print(f"  ⚠️ 解析 {xml_file} 失败 (XML 格式错误): {e}")
                continue
            except Exception as e:
                print(f"  ⚠️ 读取 {xml_file} 失败: {e}")
                continue

            # 收集所有 <testsuite> 元素
            testsuites: list[ET.Element] = []
            if root.tag == "testsuites":
                # 结构: <testsuites><testsuite>...</testsuite></testsuites>
                testsuites = root.findall("testsuite")
            elif root.tag == "testsuite":
                # 结构: <testsuite>...</testsuite> 直接作为根
                testsuites = [root]
            else:
                print(f"  ⚠️ {xml_file} 根元素非 testsuites/testsuite，跳过")
                continue

            file_test_count = 0
            for ts in testsuites:
                # 提取并聚合统计
                total_tests += int(ts.get("tests", "0"))
                total_failures += int(ts.get("failures", "0"))
                total_errors += int(ts.get("errors", "0"))
                total_skipped += int(ts.get("skipped", "0"))
                with contextlib.suppress(ValueError):
                    total_time += float(ts.get("time", "0.0"))

                # 提取所有 testcase 元素，剔除含 <skipped> 子元素的用例
                for tc in ts.findall("testcase"):
                    if tc.find("skipped") is not None:
                        continue
                    normalized_name = re.sub(r"[^0-9a-zA-Z]+", "_", tc.get("classname")) + "_" + tc.get("name")
                    tc.set("name", normalized_name)
                    all_testcases.append(tc)
                    file_test_count += 1

            if verbose:
                print(f"  📋 已合并 {xml_file}: {file_test_count} 个测试用例")

    # 构建最终的 XML 结构
    # tests 计数减去被剔除的 skipped 用例；skipped 属性归零
    merged_tests = total_tests - total_skipped
    final_root = ET.Element("testsuites")
    merged_suite = ET.SubElement(
        final_root,
        "testsuite",
        name="pytest-merged",
        tests=str(merged_tests),
        failures=str(total_failures),
        errors=str(total_errors),
        skipped="0",
        time=f"{total_time:.3f}",
    )
    for tc in all_testcases:
        merged_suite.append(tc)

    # 写入最终 XML
    try:
        final_tree = ET.ElementTree(final_root)
        ET.indent(final_tree, space="  ")  # Python 3.9+ 格式化输出
    except AttributeError:
        # Python < 3.9 没有 ET.indent，跳过格式化
        final_tree = ET.ElementTree(final_root)

    final_tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"\n✅ 测试结果已合并到: {output_path}")
    print(f"   测试总数={total_tests}, 失败={total_failures}, 错误={total_errors}, 跳过={total_skipped}, 耗时={total_time:.3f}s")
    return True


# ---------------------------------------------------------------------------
# 清理临时文件
# ---------------------------------------------------------------------------


def cleanup_temp_files(xml_files: list[str], verbose: bool = False) -> None:
    """
    清理临时生成的 XML 文件。

    参数:
        xml_files: 待清理的文件路径列表
        verbose:   是否输出详细信息
    """
    if not xml_files:
        return

    cleaned = 0
    for f in xml_files:
        try:
            if os.path.exists(f):
                os.remove(f)
                cleaned += 1
                if verbose:
                    print(f"  🗑️ 已删除: {f}")
        except OSError as e:
            print(f"  ⚠️ 删除 {f} 失败: {e}")

    print(f"🧹 清理完成: 已删除 {cleaned}/{len(xml_files)} 个临时文件")


# ---------------------------------------------------------------------------
# 摘要报告
# ---------------------------------------------------------------------------


def print_summary(results: list[TestResult], output_xml: str) -> None:
    """
    打印测试执行的摘要表格。

    参数:
        results:    所有测试运行结果
        output_xml: 合并后的 XML 文件路径
    """
    print(f"\n{'=' * 70}")
    print("                          测试执行摘要")
    print(f"{'=' * 70}")

    if not results:
        print("  (无测试运行)")
    else:
        # 表头
        print(f"  {'序号':<6}{'状态':<8}{'耗时':>10}  {'测试用例'}")
        print(f"  {'-' * 6}{'-' * 8}{'-' * 10}  {'-' * 40}")

        passed_count = 0
        failed_count = 0
        skipped_count = 0
        total_duration = 0.0

        for i, r in enumerate(results, start=1):
            status = r.status_label
            duration_str = f"{r.duration:.2f}s"
            name = r.config.display_name
            # 截断过长的名称
            if len(name) > 120:
                name = "..." + name[-117:]

            if status == "PASS":
                marker = "✅"
            elif status == "SKIP":
                marker = "⏭️"
            else:
                marker = "❌"

            # 运行期 skip 的用例不逐条列出，只计数
            if status != "SKIP":
                print(f"  {i:<6}{marker} {status:<5}{duration_str:>10}  {name}")

            if status == "PASS":
                passed_count += 1
            elif status == "SKIP":
                skipped_count += 1
            else:
                failed_count += 1
            total_duration += r.duration

        print(f"  {'-' * 6}{'-' * 8}{'-' * 10}  {'-' * 40}")
        print(
            f"  {'合计':<6}{'':8}{total_duration:>9.2f}s  "
            f"通过={passed_count}, 失败={failed_count}, "
            f"跳过={skipped_count}, 总计={len(results)}"
        )

    print(f"\n  📄 合并结果文件: {os.path.abspath(output_xml)}")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# 失败用例输出
# ---------------------------------------------------------------------------


def dump_fail_list(results: list[TestResult], output_path: str = "fail_list.json") -> None:
    """
    将失败的测试用例输出为 JSON 文件，格式与 <BOARD_TYPE>_skip.json 一致。

    参数:
        results:     所有测试运行结果
        output_path: 输出文件路径 (默认: fail_list.json)
    """
    fail_entries = []
    for r in results:
        if r.failed:
            fail_entries.append(
                {
                    "file_path": r.config.file_path,
                    "test_filter": r.config.test_filter or "",
                    "extra_args": list(r.config.extra_args),
                    "reason": "TIMEOUT" if r.timed_out else "N/A",
                }
            )

    with open(output_path, "w", encoding="utf-8") as f:
        print(f"Fail cases list:\n{fail_entries}")
        json.dump(fail_entries, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"📝 失败用例列表已保存到: {output_path}")
    print(f"   共 {len(fail_entries)} 个失败用例")
    print(f"{'=' * 70}\n")


def dump_skip_list(results: list[TestResult], output_path: str = "skip_list.json") -> None:
    """
    运行期 skip 不需要持久化——写空列表保持文件契约即可。

    参数:
        results:     所有测试运行结果
        output_path: 输出文件路径 (默认: skip_list.json)
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"📝 跳过用例列表已保存到: {output_path}（运行期 skip 不写入）")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Collection error → synthetic JUnit / TestResult
# ---------------------------------------------------------------------------

_COLLECTION_ERROR_FILTER_TAG = "<collection-error>"


def _write_collection_error_xml(err: CollectionError, index: int) -> str:
    """Materialize a CollectionError as a single-testcase JUnit XML file.

    ElementTree performs XML escaping on both attribute values and .text on
    write, so we hand it the raw traceback verbatim. The synthesized file
    plugs into the same merge_junit_xml() pipeline as normal per-case XMLs.
    """
    xml_path = f"collection_error_{index}.xml"
    root = ET.Element("testsuites")
    ts = ET.SubElement(
        root,
        "testsuite",
        name="pytest-collection-error",
        tests="1",
        failures="0",
        errors="1",
        skipped="0",
        time="0.000",
    )
    tc = ET.SubElement(
        ts,
        "testcase",
        classname=err.file_path,
        name="collection_error",
        time="0.000",
    )
    error_el = ET.SubElement(
        tc,
        "error",
        type="CollectionError",
        message=f"pytest collection error for {err.file_path}",
    )
    error_el.text = err.traceback or ""
    tree = ET.ElementTree(root)
    with contextlib.suppress(AttributeError):
        ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path


def _make_collection_error_result(err: CollectionError, xml_path: str) -> TestResult:
    """Build a synthetic TestResult so the collection error flows through the
    existing print_summary / dump_fail_list / merge pipeline unchanged.

    Uses returncode=-2 and xml_errors=1 so status_label resolves to FAIL and
    the entry counts toward has_failures at exit time.
    """
    cfg = TestConfig(
        file_path=err.file_path,
        test_filter=_COLLECTION_ERROR_FILTER_TAG,
    )
    return TestResult(
        config=cfg,
        returncode=-2,
        duration=0.0,
        xml_path=xml_path,
        stdout="",
        stderr=err.traceback,
        xml_tests=1,
        xml_failures=0,
        xml_errors=1,
        xml_skipped=0,
        timed_out=False,
    )


def _dedupe_collection_errors(errors: list[CollectionError]) -> list[CollectionError]:
    """Dedupe by normalized file_path; first occurrence wins.

    Called across per-directory scan results, so the same broken file surfaced
    from multiple target dirs (or by both header + short-summary parsers) only
    produces a single synthetic failure.
    """
    seen: set = set()
    out: list[CollectionError] = []
    for e in errors:
        key = os.path.normpath(e.file_path)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """
    主入口：解析命令行参数，运行测试，合并结果，输出摘要。

    参数:
        argv: 命令行参数列表（默认使用 sys.argv）

    返回:
        退出码（0=全部通过，1=有失败）
    """
    parser = argparse.ArgumentParser(
        description="通用 pytest 测试运行与 JUnit XML 合并框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例用法:\n"
            "  python test_tilelang.py\n"
            "  python test_tilelang.py -o result.xml --test-dir /path/to/repo\n"
            "  python test_tilelang.py --keep-temp -v\n"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="test-results.xml",
        help="合并后的 XML 输出文件路径 (默认: test-results.xml)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="保留临时 XML 文件，不在合并后删除",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="输出详细信息",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TEST_TIMEOUT_SECONDS,
        help="单个用例最长运行时间（秒），超时强制终止并判定为 FAIL (默认: 7200，即 2 小时)",
    )
    parser.add_argument(
        "--maxfail",
        type=int,
        default=0,
        help="Stop after N real failures (0=no limit)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel test workers (default 1=serial)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max test cases per directory group (0=no limit)",
    )

    args = parser.parse_args(argv)

    # Examples subdirectories to test (under examples/, which has pytest.ini)
    # pytest.ini makes rootdir=examples/ → collected paths relative to examples/
    # Add/remove entries here to control which example dirs are tested
    EXAMPLE_SUBDIRS = [
        "flash_attention",
        "gemm",
        "ppu",
        "chained_dot",
        "analyze",
        "attention_sink",
        "blocksparse_attention",
        "cast",
        "convolution",
        "deepseek_mhc",
        "deepseek_nsa",
        "deepseek_v4",
        "elementwise",
        "gdn",
        "gemm_splitk",
        "gemm_streamk",
        "gemv",
        "kda",
        "norm",
        "seer_attention",
        "topk",
    ]
    EXAMPLE_PREFIX = "examples"

    test_dirs: list[tuple[str, str]] = [(f"{EXAMPLE_PREFIX}/{d}", EXAMPLE_PREFIX) for d in EXAMPLE_SUBDIRS]

    # testing/ sub-directories to scan (skip non-PPU targets: cpu, metal, webgpu)
    TESTING_SKIP_DIRS = {"cpu", "metal", "webgpu"}
    testing_subdirs = sorted(
        d for d in os.listdir("testing/python") if os.path.isdir(os.path.join("testing/python", d)) and d not in TESTING_SKIP_DIRS
    )
    test_dirs += [(f"testing/python/{d}", ".") for d in testing_subdirs]
    test_configs: list[TestConfig] = []
    all_collection_errors: list[CollectionError] = []
    for dir_tuple in test_dirs:
        # scan_cases 已完成 skip 过滤；--limit 在过滤之后、调度之前对每个
        # 目录组做截断（take first N per group），既能覆盖多个目录又快速跑通。
        group, group_errors = scan_cases(dir_tuple)
        if args.limit > 0 and len(group) > args.limit:
            print(f"✂️ 目录组 {dir_tuple[0]}: 应用 --limit={args.limit}，保留前 {args.limit} 个用例（收集 {len(group)} 个）")
            group = group[: args.limit]
        test_configs.extend(group)
        all_collection_errors.extend(group_errors)

    # Deduplicate collection errors before synthesizing failure records.
    # Same file may surface from multiple scan groups or from both the
    # ERROR-block parser and the short-summary parser.
    all_collection_errors = _dedupe_collection_errors(all_collection_errors)
    if all_collection_errors:
        print(f"⚠️ 检测到 {len(all_collection_errors)} 个 pytest collection 错误，将作为失败写入 JUnit 与 fail_list")
        for e in all_collection_errors:
            print(f"   ✗ {e.file_path}")

    with open("tests.json", "w") as f:
        # tests.json 只描述实际待执行的测试；collection 错误通过合成 JUnit /
        # fail_list 单独承载，避免下游消费者把不可执行项当成正常调度目标。
        json.dump([asdict(t) for t in test_configs], f, indent=2)

    if not test_configs:
        print("⚠️ 没有配置任何测试用例")

    print(f"📋 共 {len(test_configs)} 个测试用例待运行")
    print(f"📄 输出文件: {args.output}")
    workers = max(1, args.workers)
    print(
        f"🧵 并行 worker 数: {workers}"
        + (f" | --maxfail={args.maxfail}" if args.maxfail > 0 else "")
        + (f" | --limit={args.limit}/目录组" if args.limit > 0 else "")
    )

    # 每个用例复用 run_single_test，其内部已用 Popen + os.setsid 进程组 +
    # timeout + SIGKILL + 独立 results_{idx}.xml 保证隔离；这里仅在其外层
    # 用线程池并发调度。线程只是阻塞等待 communicate，无需跨进程 pickle。
    results: list[TestResult] = []
    results_lock = threading.Lock()
    # 线程安全的“真实失败”计数，用于 --maxfail 早停
    fail_lock = threading.Lock()
    fail_count = 0
    stop_event = threading.Event()

    # 将 collection 错误先写入结果集，以保证即使后续调度或进程异常中断也
    # 不会丢失“无法收集”的失败信号。保存 xml_path 以便后续合并。
    collection_error_xml_paths: list[str] = []
    for c_idx, c_err in enumerate(all_collection_errors):
        xml_path = _write_collection_error_xml(c_err, c_idx)
        collection_error_xml_paths.append(xml_path)
        synth_result = _make_collection_error_result(c_err, xml_path)
        results.append(synth_result)
        if args.maxfail > 0:
            fail_count += 1
            if fail_count >= args.maxfail:
                stop_event.set()

    def _worker(idx: int, cfg: TestConfig) -> None:
        nonlocal fail_count
        # maxfail 已触发时，尚未开始的用例直接跳过（不再启动新子进程）
        if stop_event.is_set():
            return
        result = run_single_test(cfg, idx, verbose=args.verbose, timeout=args.timeout)
        with results_lock:
            results.append(result)
        # 沿用现有 TestResult.failed 口径统计真实失败（不含 skip）
        if result.failed and args.maxfail > 0:
            with fail_lock:
                fail_count += 1
                if fail_count >= args.maxfail:
                    stop_event.set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # idx 唯一：以 enumerate 序号绑定每个用例，保证 results_{idx}.xml 不冲突
        future_map = {executor.submit(_worker, idx, cfg): idx for idx, cfg in enumerate(test_configs)}
        for future in concurrent.futures.as_completed(future_map):
            # 触发早停后，取消尚未开始的任务；已提交/在跑的照常完成
            if stop_event.is_set():
                for pending in future_map:
                    pending.cancel()
            # 早停取消的未开始任务会抛 CancelledError，属正常流程需忽略，
            # 以便继续合并 JUnit、输出结果；其余异常仍向上传播暴露真实 worker 错误
            try:
                future.result()
            except concurrent.futures.CancelledError:
                continue

    if args.maxfail > 0 and stop_event.is_set():
        print(f"\n🛑 已达到 --maxfail={args.maxfail} 真实失败阈值，提前停止调度新用例")

    xml_files: list[str] = [r.xml_path for r in results if r.xml_path is not None]

    print(f"\n{'─' * 70}")
    print("正在合并 JUnit XML 结果...")
    merge_junit_xml(xml_files, args.output, verbose=args.verbose)

    if not args.keep_temp:
        cleanup_temp_files(xml_files, verbose=args.verbose)
    else:
        print(f"ℹ️ 保留临时文件 (--keep-temp): {xml_files}")

    print_summary(results, args.output)

    # 输出失败用例列表
    dump_fail_list(results)

    # 输出被 pytest 跳过的用例列表（含部分 skip 与全 skip），避免它们在产物里不可见
    dump_skip_list(results)

    # Only actual failures count — skipped tests are not failures
    has_failures = any(r.failed for r in results)
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
