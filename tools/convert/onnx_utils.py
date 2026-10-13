"""Small ONNX helpers shared by the export scripts."""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def compress_fp16(src, dst, keep_fp32=False, min_size=1024):
    """Store large float32 initializers as float16 and cast them back to float32
    at load time.  ONNX Runtime folds the Cast nodes when the session is
    created, so inference still runs in float32 while the file is half as big."""
    model = onnx.load(src)
    if keep_fp32:
        onnx.save(model, dst)
        return
    graph = model.graph
    new_inits, casts = [], []
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= min_size:
            half_name = init.name + "__fp16"
            new_inits.append(numpy_helper.from_array(arr.astype(np.float16), half_name))
            casts.append(helper.make_node("Cast", [half_name], [init.name], to=TensorProto.FLOAT))
        else:
            new_inits.append(init)
    del graph.initializer[:]
    graph.initializer.extend(new_inits)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, dst)
