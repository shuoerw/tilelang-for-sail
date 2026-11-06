"""End-to-end codegen tests for InjectAIUSyncBarrier pass.

Verifies that the AIU sync barrier injection pass produces the expected
cp_async_commit / cp_async_wait patterns in the generated CUDA source code
under various kernel configurations.

Scenarios covered:
  1. stage=0 basic GEMM: commit + wait presence
  2. Prefetch pattern: wrap-around wait with N > 0
  3. stage>0 pipelined: AIU lower works but barriers come from software pipeline
  4. All buffers consumed together: wait<0>
  5. No-prefetch pattern: pre-loop load with wait before loop
  6. Dynamic-extent pipelined (GQA-style): no duplicate prologue commits
  7. IfThenElse branch-local aiu_load: commit shares the branch scope
"""

import re

import tilelang
import tilelang.ppu.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey

# Global matrix dimensions
M = N = K = 512
dtype = T.float16
accum_dtype = T.float32


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------


def _stage0_gemm_kernel(block_M=128, block_N=128, block_K=64):
    """Standard GEMM with Pipelined(num_stages=0) for AIU sync barrier injection."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared, prefer_instruction="aiu")
                T.copy(B[ko * block_K, bx * block_N], B_shared, prefer_instruction="aiu")
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _prefetch_kernel(block_M=128, block_N=128, block_K=64):
    """Kernel with pre-loop prefetch of H, and multiple loads inside the loop.

    This pattern forces wrap-around wait computation to account for
    new commits between the prefetched load and its consumption point.
    """
    num_iters = T.ceildiv(K, block_K)

    @T.prim_func
    def main(
        H: T.Tensor((M, K), dtype),
        V: T.Tensor((K, N), dtype),
        K_buf: T.Tensor((K, N), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            h_shared = T.alloc_shared((block_M, block_K), dtype)
            v_shared = T.alloc_shared((block_K, block_N), dtype)
            k_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(out_local)
            # Pre-loop prefetch h
            T.copy(H[by * block_M, 0], h_shared, prefer_instruction="aiu")

            for i in T.Pipelined(num_iters, num_stages=0):
                # Load v, k for current iteration
                T.copy(V[i * block_K, bx * block_N], v_shared, prefer_instruction="aiu")
                T.copy(K_buf[i * block_K, bx * block_N], k_shared, prefer_instruction="aiu")
                # Consume h (from pre-loop or previous iteration reload)
                T.gemm(h_shared, v_shared, out_local)
                T.gemm(h_shared, k_shared, out_local)

            T.copy(out_local, Out[by * block_M, bx * block_N])

    return main


def _pipelined_gemm_kernel(block_M=128, block_N=128, block_K=64, num_stages=3):
    """Standard pipelined GEMM with num_stages > 0."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared, prefer_instruction="aiu")
                T.copy(B[ko * block_K, bx * block_N], B_shared, prefer_instruction="aiu")
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _fp8_aiu_gemm_kernel(num_stages, dtype=T.float8_e4m3fn, block_k=64):
    """FP8 GEMM used to cover AIU b8 synchronization paths."""

    @T.prim_func
    def main(
        A: T.Tensor((128, K), dtype),
        B: T.Tensor((128, K), dtype),
        C: T.Tensor((128, 128), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, block_k), dtype)
            B_shared = T.alloc_shared((128, block_k), dtype)
            C_local = T.alloc_fragment((128, 128), T.float32)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                T.copy(A[0, ko * block_k], A_shared, prefer_instruction="aiu")
                T.copy(B[0, ko * block_k], B_shared, prefer_instruction="aiu")
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C)

    return main


def _no_prefetch_kernel(block_M=128, block_N=128, block_K=64):
    """Kernel that loads H before the loop and consumes it inside without reload.

    The wait must appear before the loop since H is not reloaded internally.
    """
    num_iters = T.ceildiv(K, block_K)

    @T.prim_func
    def main(
        H: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            h_shared = T.alloc_shared((block_M, block_K), dtype)
            b_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(out_local)
            # Pre-loop load (no reload inside loop)
            T.copy(H[by * block_M, 0], h_shared, prefer_instruction="aiu")

            for i in T.Pipelined(num_iters, num_stages=0):
                T.copy(B[i * block_K, bx * block_N], b_shared, prefer_instruction="aiu")
                T.gemm(h_shared, b_shared, out_local)

            T.copy(out_local, Out[by * block_M, bx * block_N])

    return main


def _dynamic_extent_kv_kernel(block_M=128, block_N=128, block_K=64, num_stages=2):
    """Minimal GQA-decode-style kernel with a dynamic (runtime) loop extent.

    Q is loaded before the loop; K/V tiles are loaded and consumed
    alternately inside a num_stages > 0 pipelined loop whose extent is a
    runtime scalar. The unprovable prologue predicates make
    InjectSoftwarePipeline emit already-lowered (call-form) commits inside
    predicated IfThenElse groups in the kernel prologue, which is exactly
    where InjectAIUSyncBarrier used to re-inject duplicate commits.
    """

    @T.prim_func
    def main(
        Q: T.Tensor((M, block_K), dtype),
        K_mat: T.Tensor((K, block_K), dtype),
        V_mat: T.Tensor((K, block_K), dtype),
        Out: T.Tensor((M, block_K), dtype),
        real_k: T.int32,
    ):
        with T.Kernel(1, T.ceildiv(M, block_M), threads=128) as (bx, by):
            Q_shared = T.alloc_shared((block_M, block_K), dtype)
            K_shared = T.alloc_shared((block_N, block_K), dtype)
            V_shared = T.alloc_shared((block_N, block_K), dtype)
            S_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            S_cast = T.alloc_fragment((block_M, block_N), dtype)
            O_local = T.alloc_fragment((block_M, block_K), accum_dtype)

            T.copy(Q[by * block_M, 0], Q_shared, prefer_instruction="aiu")
            T.clear(O_local)
            loop_range = T.ceildiv(real_k, block_N)
            for ko in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K_mat[ko * block_N, 0], K_shared, prefer_instruction="aiu")
                T.clear(S_local)
                T.gemm(
                    Q_shared,
                    K_shared,
                    S_local,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(S_local, S_cast)
                T.copy(V_mat[ko * block_N, 0], V_shared, prefer_instruction="aiu")
                T.gemm(
                    S_cast,
                    V_shared,
                    O_local,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            T.copy(O_local, Out[by * block_M, 0])

    return main


def _if_else_aiu_load_kernel(block_M=128, block_N=128, block_K=64):
    """Kernel with aiu_load statements inside both branches of a dynamic
    IfThenElse (no pipeline loop).

    The commit for a branch-local aiu_load must be injected into the same
    conditional scope as the load: never emitted on the unconditional path
    (phantom commit) and never duplicated.
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
        flag: T.int32,
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            # Both branches load a different A tile into the same shared
            # buffer, so the aiu_load (and its commit) must stay in-branch.
            if flag > 0:
                T.copy(A[by * block_M, 0], A_shared, prefer_instruction="aiu")
            else:
                T.copy(A[by * block_M, block_K], A_shared, prefer_instruction="aiu")
            T.copy(B[0, bx * block_N], B_shared, prefer_instruction="aiu")
            T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


# ---------------------------------------------------------------------------
# Helper: compile with AIU enabled
# ---------------------------------------------------------------------------


def _compile_with_aiu(func):
    """Compile a kernel with AIU lowering enabled and return the CUDA source."""
    kernel = tilelang.compile(
        func,
        out_idx=[-1],
        pass_configs={
            PassConfigKey.TL_DISABLE_AIU_LOWER: False,
        },
    )
    return kernel.get_kernel_source()


# ---------------------------------------------------------------------------
# Helpers: CUDA source inspection
# ---------------------------------------------------------------------------


def _extract_braced_block(source, open_idx):
    """Return (body, close_idx) of the brace block starting at open_idx.

    ``body`` excludes the outer braces. Returns (None, open_idx) when the
    braces are unbalanced.
    """
    depth = 0
    for i in range(open_idx, len(source)):
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[open_idx + 1 : i], i
    return None, open_idx


def _conditional_blocks(source):
    """Return body texts of ``if (...) { ... }`` / ``else { ... }`` blocks.

    Uses balanced paren/brace scanning over the generated CUDA source so the
    commit placement relative to branch-local aiu_loads can be checked.
    Both then- and else-branch bodies are included.

    Note: the scanner is not aware of parens/braces inside string literals
    or comments; it is only safe for the controlled PPU codegen output
    (extern call form, IfThenElse always printed with braces). Do not reuse
    it on arbitrary/uncontrolled source code.
    """
    blocks = []
    for match in re.finditer(r"\bif\s*\(", source):
        # Balance-match the condition parentheses.
        i = match.end() - 1
        depth = 0
        while i < len(source):
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        # Skip whitespace up to the then-block opening brace.
        j = i + 1
        while j < len(source) and source[j].isspace():
            j += 1
        if j >= len(source) or source[j] != "{":
            continue
        then_body, close_idx = _extract_braced_block(source, j)
        if then_body is None:
            continue
        blocks.append(then_body)
        # Look ahead for an optional matching ``else {``.
        e = close_idx + 1
        while e < len(source) and source[e].isspace():
            e += 1
        if not source.startswith("else", e):
            continue
        e += len("else")
        while e < len(source) and source[e].isspace():
            e += 1
        if e < len(source) and source[e] == "{":
            else_body, _ = _extract_braced_block(source, e)
            if else_body is not None:
                blocks.append(else_body)
    return blocks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_stage0_gemm_has_commit_and_wait():
    """stage=0 GEMM should generate aiu_load + cp_async_commit + cp_async_wait."""
    source = _compile_with_aiu(_stage0_gemm_kernel())

    assert "tl::aiu_load" in source, "Expected tl::aiu_load in generated CUDA source"
    assert "tl::cp_async_commit()" in source, "Expected tl::cp_async_commit() in generated CUDA source"
    assert "tl::cp_async_wait" in source, "Expected tl::cp_async_wait in generated CUDA source"


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_prefetch_pattern_wait_gt_zero():
    """When a loop has pre-loop prefetch + loop-body loads, the wrap-around
    wait should have N > 0 (allowing newer groups to remain in-flight)."""
    source = _compile_with_aiu(_prefetch_kernel())

    assert "tl::aiu_load" in source, "Expected tl::aiu_load in generated CUDA source"
    assert "tl::cp_async_commit()" in source, "Expected tl::cp_async_commit() in generated CUDA source"

    # Find all cp_async_wait<N> occurrences and check that at least one has N > 0
    wait_pattern = re.findall(r"tl::cp_async_wait<(\d+)>", source)
    assert len(wait_pattern) > 0, "Expected at least one tl::cp_async_wait<N> in generated CUDA source"
    has_nonzero_wait = any(int(n) > 0 for n in wait_pattern)
    assert has_nonzero_wait, f"Expected at least one cp_async_wait<N> with N > 0 for prefetch pattern, but found only: {wait_pattern}"


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_stage_gt0_no_aiu_sync_barrier():
    """stage>0 loops (pipelined) should NOT get AIU sync barrier treatment.

    The commit/wait in pipelined mode comes from InjectSoftwarePipeline,
    not InjectAIUSyncBarrier. This test verifies pipelined compilation works
    correctly with AIU enabled.
    """
    source = _compile_with_aiu(_pipelined_gemm_kernel(num_stages=3))

    # AIU lower should still be active (aiu_load present)
    assert "tl::aiu_load" in source, "Expected tl::aiu_load in generated CUDA source for pipelined GEMM"
    # Pipelined mode should still produce valid code with async operations
    assert "tl::cp_async_commit()" in source and "tl::cp_async_wait" in source, (
        "Expected both cp_async_commit and cp_async_wait in pipelined GEMM source"
    )


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_fp8_stage0_has_b8_aiu_sync_barrier():
    source = _compile_with_aiu(_fp8_aiu_gemm_kernel(num_stages=0))

    assert "tl::aiu_load" in source
    assert "tl::cp_async_commit()" in source
    assert "tl::cp_async_wait" in source


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_fp8_pipeline_has_b8_aiu_sync_operations():
    source = _compile_with_aiu(_fp8_aiu_gemm_kernel(num_stages=3))

    assert "tl::aiu_load" in source
    assert "tl::cp_async_commit()" in source
    assert "tl::cp_async_wait" in source


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_fp8_e5m2_stage0_has_b8_aiu_sync_barrier():
    source = _compile_with_aiu(_fp8_aiu_gemm_kernel(num_stages=0, dtype=T.float8_e5m2))

    assert "tl::aiu_load" in source
    assert "tl::cp_async_commit()" in source
    assert "tl::cp_async_wait" in source


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_fp8_e5m2_pipeline_has_b8_aiu_sync_operations():
    source = _compile_with_aiu(_fp8_aiu_gemm_kernel(num_stages=3, dtype=T.float8_e5m2))

    assert "tl::aiu_load" in source
    assert "tl::cp_async_commit()" in source
    assert "tl::cp_async_wait" in source


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_all_buffers_consumed_together_wait_zero():
    """When all loaded buffers are consumed at the same point (the newest),
    wait<0> is correct (must wait for all)."""
    source = _compile_with_aiu(_stage0_gemm_kernel())

    assert "tl::aiu_load" in source, "Expected tl::aiu_load in generated CUDA source"
    # In a standard stage=0 GEMM where A and B are loaded and immediately
    # consumed by gemm, all groups must be waited for → wait<0>
    assert "tl::cp_async_wait<0>" in source, "Expected tl::cp_async_wait<0> when all buffers are consumed together"


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_no_prefetch_wait_at_loop():
    """When pre-loop loads are consumed inside a loop without internal reload,
    wait should appear before the loop (no wrap-around possible)."""
    source = _compile_with_aiu(_no_prefetch_kernel())

    assert "tl::aiu_load" in source, "Expected tl::aiu_load in generated CUDA source"
    assert "tl::cp_async_commit()" in source, "Expected tl::cp_async_commit() in generated CUDA source"
    # Without prefetch reload inside the loop, wait<0> should be used
    # to ensure the pre-loop load is complete before entering the loop
    assert "tl::cp_async_wait<0>" in source, "Expected tl::cp_async_wait<0> before loop when no internal prefetch"


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_dynamic_extent_pipeline_no_duplicate_commit():
    """Prologue commits already lowered by InjectSoftwarePipeline (call form,
    guarded by IfThenElse) must not be re-committed by InjectAIUSyncBarrier.

    Every aiu_load site must map to exactly one commit site: the total number
    of cp_async_commit calls must equal the number of aiu_load calls.

    Note: this equality assumes InjectSoftwarePipeline forms one commit
    group per copy (rewriter.cc tags each async copy with its own group when
    a consumer sits between consecutive loads, as in this kernel). If the
    grouping strategy ever merges multiple loads into a shared commit group,
    this count assertion must be updated accordingly.
    """
    source = _compile_with_aiu(_dynamic_extent_kv_kernel())

    assert "tl::aiu_load" in source, "Expected tl::aiu_load in generated CUDA source"
    num_loads = source.count("tl::aiu_load")
    num_commits = source.count("tl::cp_async_commit()")
    assert num_commits == num_loads, (
        f"Expected exactly one tl::cp_async_commit() per tl::aiu_load, but "
        f"found {num_commits} commits for {num_loads} aiu_load calls "
        f"(duplicate commit injection)"
    )
    # The dynamic extent must actually materialize the predicated prologue
    # commits (call form inside IfThenElse); otherwise the count check above
    # would pass vacuously.
    has_guarded_commit = any("tl::aiu_load" in block and "tl::cp_async_commit()" in block for block in _conditional_blocks(source))
    assert has_guarded_commit, (
        "Expected a predicated block containing both tl::aiu_load and tl::cp_async_commit() (call-form commit inside IfThenElse)"
    )


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_if_branch_aiu_load_commit_shares_branch_scope():
    """aiu_loads inside IfThenElse branches must get their commit injected
    into the same conditional scope: no phantom commit on the unconditional
    path and no duplicates."""
    source = _compile_with_aiu(_if_else_aiu_load_kernel())

    assert "tl::aiu_load" in source, "Expected tl::aiu_load in generated CUDA source"
    # No duplicates and no phantom commits: one commit site per aiu_load site.
    num_loads = source.count("tl::aiu_load")
    num_commits = source.count("tl::cp_async_commit()")
    assert num_commits == num_loads, (
        f"Expected exactly one tl::cp_async_commit() per tl::aiu_load, but found {num_commits} commits for {num_loads} aiu_load calls"
    )
    # The commit must live inside the same conditional branch as the
    # branch-local aiu_load, never on the unconditional path around the if.
    branch_blocks = [block for block in _conditional_blocks(source) if "tl::aiu_load" in block]
    assert branch_blocks, "Expected an if/else branch containing tl::aiu_load in generated CUDA source"
    for block in branch_blocks:
        assert "tl::cp_async_commit()" in block, (
            "Expected tl::cp_async_commit() inside the same conditional branch as the branch-local aiu_load"
        )


if __name__ == "__main__":
    tilelang.testing.main()
