import onnx
from onnx2torch import convert
from onnx import shape_inference


model = onnx.load('./exported/mobilenet_nms.onnx')

selected_indices = onnx.helper.make_tensor_value_info('selected_indices', onnx.TensorProto.INT64, [None, 3])
model.graph.output.pop(0)
model.graph.output.append(selected_indices)

# idx0 = onnx.helper.make_tensor('gather_idx_box', onnx.TensorProto.INT64, [2], [0, 2])
# model.graph.initializer.extend([idx0])

# gather_idx_box = onnx.helper.make_node(
#     "Gather",
#     inputs=["selected_indices", "gather_idx_box"],
#     outputs=["box_idx"],
#     axis=1,
# )

# gather_batch_cls = onnx.helper.make_node(
#     "GatherND",
#     inputs=["cls_t", "selected_indices"],
#     outputs=["raw_out_cls"],
# )

# gather_batch_box = onnx.helper.make_node(
#     "GatherND",
#     inputs=["526", "box_idx"],
#     outputs=["out_box"],
# )

# unsqueeze_cls_axes = onnx.helper.make_tensor('unsqueeze_cls_axes', onnx.TensorProto.INT64, [1], [1])
# model.graph.initializer.extend([unsqueeze_cls_axes])
# unsqueeze_cls = onnx.helper.make_node(
#     "Unsqueeze",
#     inputs=["raw_out_cls", "unsqueeze_cls_axes"],
#     outputs=["out_cls"],
# )

# concat_boxes = onnx.helper.make_node(
#     "Concat",
#     inputs=["out_box", "out_cls"],
#     outputs=["raw_out"],
#     axis=1,
# )

# unsqueeze_out_axes = onnx.helper.make_tensor('unsqueeze_out_axes', onnx.TensorProto.INT64, [1], [0])
# model.graph.initializer.extend([unsqueeze_out_axes])
# unsqueeze_out = onnx.helper.make_node(
#     "Unsqueeze",
#     inputs=["raw_out", "unsqueeze_out_axes"],
#     outputs=["output"],
# )

# out = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, None, 5])

# model.graph.node.extend([gather_idx_box, gather_batch_cls, gather_batch_box, unsqueeze_cls, concat_boxes, unsqueeze_out])
# model.graph.output.pop(0)
# model.graph.output.extend([out])

# onnx.checker.check_model(model, full_check=True)
# inf_model = shape_inference.infer_shapes(model)
# onnx.save(inf_model, './exported/mobilenet_nms_test.onnx')

torch_model = convert(model)
print(torch_model)