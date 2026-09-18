"""Target validation for scoped scalar atomic operations."""

from tvm import IRModule, tirx
from tvm.ir import Op
from tvm.target import Target


_SCOPE_ARG_INDEX = {
    "tl.atomic_add_elem_op": 3,
    "tl.atomic_add_ret_elem_op": 3,
    "tl.atomic_load_elem_op": 2,
    "tl.atomic_store_elem_op": 3,
}


def validate_atomic_scopes(mod: IRModule, target: Target) -> None:
    """Reject unsupported scopes before backend lowering or code generation."""

    def visit(node):
        if not isinstance(node, tirx.Call) or not isinstance(node.op, Op):
            return
        scope_index = _SCOPE_ARG_INDEX.get(node.op.name)
        if scope_index is None:
            return
        min_args = 2 if node.op.name in ("tl.atomic_add_elem_op", "tl.atomic_add_ret_elem_op") else scope_index
        if not min_args <= len(node.args) <= scope_index + 1:
            raise ValueError(f"{node.op.name} expects {min_args} to {scope_index + 1} arguments")
        if len(node.args) <= scope_index:
            return  # Legacy calls use device scope.
        scope = node.args[scope_index]
        if not isinstance(scope, tirx.StringImm) or scope.value not in ("block", "device", "system"):
            raise ValueError(f"{node.op.name} scope must be a constant string: block, device or system")
        if scope.value != "device" and target.kind.name != "ppu":
            raise ValueError(f"{node.op.name}: {scope.value} scope is not implemented for target '{target.kind.name}'")

    for func in mod.functions.values():
        if isinstance(func, tirx.PrimFunc):
            tirx.stmt_functor.post_order_visit(func.body, visit)
