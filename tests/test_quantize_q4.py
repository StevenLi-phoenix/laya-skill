"""4-bit MatMulNBits quantization of laya-ts split exports (assets/webgpu/tools/quantize_q4.py)."""
import importlib.util
import json

import numpy as np
import pytest

from conftest import SKILL

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

TOOL = SKILL / "assets" / "webgpu" / "tools" / "quantize_q4.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("quantize_q4", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_matmul_model(path, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((64, 32)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])], "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 32])],
        initializer=[numpy_helper.from_array(w, "w")])
    # IR 9: new onnx releases default to an IR version older onnxruntime builds refuse to load
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
    onnx.save_model(model, str(path))
    return w


def _run(path, x):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]


@pytest.mark.parametrize("with_head", [False, True])
def test_quantizes_matmuls_and_copies_the_rest(tmp_path, with_head):
    src, dst = tmp_path / "fp32", tmp_path / "q4"
    src.mkdir()
    _tiny_matmul_model(src / "encoder.onnx", 0)
    _tiny_matmul_model(src / "head.onnx", 1)
    (src / "tokenizer.json").write_text("{}")
    (src / "rl_agent_config.json").write_text(json.dumps({"hidden": 64}))

    tool = _load_tool()
    tool.quantize_encoder(src, dst.mkdir() or dst, block_size=32)
    if with_head:
        tool.quantize_encoder(src, dst, block_size=32, name="head")
    tool.copy_rest(src, dst, head_quantized=with_head)

    for name, quantized in (("encoder", True), ("head", with_head)):
        ops = {n.op_type for n in onnx.load(str(dst / f"{name}.onnx")).graph.node}
        assert ("MatMulNBits" in ops) == quantized, (name, ops)
    assert (dst / "tokenizer.json").exists() and (dst / "rl_agent_config.json").exists()

    x = np.random.default_rng(2).standard_normal((4, 64)).astype(np.float32)
    ref, got = _run(src / "encoder.onnx", x), _run(dst / "encoder.onnx", x)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 0.15, f"4-bit error too large: {rel:.3f}"
    assert not np.array_equal(got, ref)  # it really was quantized
