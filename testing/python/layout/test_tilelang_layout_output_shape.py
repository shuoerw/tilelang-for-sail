"""Physical layout extents must cover nonzero address offsets."""

import pytest

from tilelang import Layout, tvm


@pytest.mark.parametrize("offset", [0, 32, 128])
@pytest.mark.parametrize("bitwise", [False, True])
def test_output_shape_includes_address_prefix(offset, bitwise):
    layout = Layout(
        (32,),
        lambda i: (tvm.tirx.bitwise_xor(i, 7) if bitwise else i) + offset,
    )
    assert list(map(int, layout.get_output_shape())) == [32 + offset]


def test_each_output_dimension_includes_its_prefix():
    layout = Layout((8, 32), lambda i, j: [i + 8, tvm.tirx.bitwise_xor(j, 7) + 32])
    assert list(map(int, layout.get_output_shape())) == [16, 64]
