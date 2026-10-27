"""End-to-end checks for packed and scalar PPU subword shuffles."""

import pytest
import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing


LANES = 32
BYTES_PER_WORD = 4


def _make_kernel(dtype, source_lane_count, conditional):
    compute_dtype = T.int32 if dtype == T.int8 else T.float32

    if conditional:

        @T.prim_func
        def main(
            a: T.Tensor((LANES, BYTES_PER_WORD), dtype),
            b: T.Tensor((LANES, BYTES_PER_WORD), dtype),
            output: T.Tensor((LANES, BYTES_PER_WORD), dtype),
        ):
            with T.Kernel(1, threads=LANES):
                lane = T.get_lane_idx()
                values = T.alloc_local((BYTES_PER_WORD,), dtype)
                for byte in T.unroll(BYTES_PER_WORD):
                    group_base = lane // 4 * 4
                    true_lane = group_base + byte % source_lane_count
                    false_lane = group_base + (byte + 1) % source_lane_count
                    true_value = T.shfl_sync(T.cast(a[lane, byte], compute_dtype), true_lane)
                    false_value = T.shfl_sync(T.cast(b[lane, byte], compute_dtype), false_lane)
                    values[byte] = T.cast(
                        T.if_then_else(lane % 2 == 0, true_value, false_value),
                        dtype,
                    )
                for byte in T.unroll(BYTES_PER_WORD):
                    output[lane, byte] = values[byte]

    else:

        @T.prim_func
        def main(
            a: T.Tensor((LANES, BYTES_PER_WORD), dtype),
            b: T.Tensor((LANES, BYTES_PER_WORD), dtype),
            output: T.Tensor((LANES, BYTES_PER_WORD), dtype),
        ):
            with T.Kernel(1, threads=LANES):
                lane = T.get_lane_idx()
                values = T.alloc_local((BYTES_PER_WORD,), dtype)
                for byte in T.unroll(BYTES_PER_WORD):
                    source_lane = lane // 4 * 4 + byte % source_lane_count
                    values[byte] = T.cast(
                        T.shfl_sync(T.cast(a[lane, byte], compute_dtype), source_lane),
                        dtype,
                    )
                for byte in T.unroll(BYTES_PER_WORD):
                    output[lane, byte] = values[byte]

    return main


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize(
    "dtype,source_lane_count,packed",
    [(T.float8_e4m3fn, lanes, True) for lanes in (1, 2, 3, 4)]
    + [(T.int8, lanes, lanes <= 2) for lanes in (1, 2, 3, 4)]
    + [(T.float8_e5m2, 2, False)],
)
def test_subword_source_lane_gather(dtype, source_lane_count, packed, conditional):
    kernel = tilelang.compile(_make_kernel(dtype, source_lane_count, conditional), out_idx=[2])
    source = kernel.get_kernel_source()
    expected_shuffles = source_lane_count * (2 if conditional else 1) if packed else (2 if conditional else 1)
    assert source.count("__shfl_sync") == expected_shuffles
    assert source.count("__byte_perm") == ((source_lane_count - 1) * (2 if conditional else 1) if packed else 0)

    torch_dtype = dtype.as_torch()
    data = torch.arange(LANES * BYTES_PER_WORD, device="cuda", dtype=torch.int16)
    a = ((data % 19) - 9).reshape(LANES, BYTES_PER_WORD).to(torch_dtype)
    b = ((data % 17) - 8).reshape(LANES, BYTES_PER_WORD).to(torch_dtype)
    actual = kernel(a, b)

    expected = torch.empty_like(a)
    for lane in range(LANES):
        for byte in range(BYTES_PER_WORD):
            group_base = lane // 4 * 4
            if conditional and lane % 2:
                source_lane = group_base + (byte + 1) % source_lane_count
                expected[lane, byte] = b[source_lane, byte]
            else:
                source_lane = group_base + byte % source_lane_count
                expected[lane, byte] = a[source_lane, byte]
    torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
