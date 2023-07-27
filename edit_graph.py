import onnx
from onnx import shape_inference


model = onnx.load('./exported/mobilenetv2.onnx')

iou_threshold = onnx.helper.make_tensor('iou_threshold', onnx.TensorProto.FLOAT, [1], [0.45])
score_threshold = onnx.helper.make_tensor('score_threshold', onnx.TensorProto.FLOAT, [1], [0.01])
max_output_boxes_per_class = onnx.helper.make_tensor('max_output_boxes_per_class', onnx.TensorProto.INT64, [1], [5])

model.graph.initializer.extend([iou_threshold, score_threshold, max_output_boxes_per_class])
nms = onnx.helper.make_node(
    "NonMaxSuppression",
    inputs=[
        "970",
        "cls_t",
        "max_output_boxes_per_class",
        "iou_threshold",
        "score_threshold",
    ],
    outputs=["selected_indices"],
    center_point_box=1,
)

cls_t = onnx.helper.make_node(
    "Transpose",
    inputs=["933"],
    outputs=["cls_t"],
    perm=(0, 2, 1),
)
model.graph.node.extend([cls_t, nms])

idx0 = onnx.helper.make_tensor('gather_idx_box', onnx.TensorProto.INT64, [2], [0, 2])
idx1 = onnx.helper.make_tensor('gather_idx_cls', onnx.TensorProto.INT64, [2], [0, 1])
model.graph.initializer.extend([idx0, idx1])

gather_idx_box = onnx.helper.make_node(
    "Gather",
    inputs=["selected_indices", "gather_idx_box"],
    outputs=["box_idx"],
    axis=1,
)

gather_batch_cls = onnx.helper.make_node(
    "GatherND",
    inputs=["cls_t", "selected_indices"],
    outputs=["raw_out_cls"],
)

gather_batch_box = onnx.helper.make_node(
    "GatherND",
    inputs=["970", "box_idx"],
    outputs=["out_box"],
)

gather_idx_cls = onnx.helper.make_node(
    "Gather",
    inputs=["selected_indices", "gather_idx_cls"],
    outputs=["cls_idx_raw"],
    axis=1,
)

cast_cls = onnx.helper.make_node(
    'Cast',
    inputs=["cls_idx_raw"],
    outputs=['cls_idx'],
    to=onnx.TensorProto.FLOAT,
)

unsqueeze_cls_axes = onnx.helper.make_tensor('unsqueeze_cls_axes', onnx.TensorProto.INT64, [1], [1])
model.graph.initializer.extend([unsqueeze_cls_axes])
unsqueeze_cls = onnx.helper.make_node(
    "Unsqueeze",
    inputs=["raw_out_cls", "unsqueeze_cls_axes"],
    outputs=["out_cls"],
)

concat_boxes = onnx.helper.make_node(
    "Concat",
    inputs=["cls_idx", "out_cls", "out_box"],
    outputs=["raw_out"],
    axis=1,
)

unsqueeze_out_axes = onnx.helper.make_tensor('unsqueeze_out_axes', onnx.TensorProto.INT64, [2], [0, 1])
model.graph.initializer.extend([unsqueeze_out_axes])
unsqueeze_out = onnx.helper.make_node(
    "Unsqueeze",
    inputs=["raw_out", "unsqueeze_out_axes"],
    outputs=["output"],
)

out = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, 1, None, 7])

model.graph.node.extend([gather_idx_box, gather_batch_cls, gather_batch_box, unsqueeze_cls, gather_idx_cls, cast_cls, concat_boxes, unsqueeze_out])
model.graph.output.pop(0)
model.graph.output.pop(1)
model.graph.output.extend([out])

onnx.checker.check_model(model, full_check=True)
inf_model = shape_inference.infer_shapes(model)
onnx.save(inf_model, './exported/mobilenetv2_test.onnx')