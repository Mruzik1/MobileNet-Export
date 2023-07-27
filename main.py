import sys
sys.path.append('./mobilenets-ssd-pytorch')

import torch
import onnx
import onnxsim
import blobconverter
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite


# model = create_mobilenetv2_ssd_lite(21, is_test=True)
# model.load('exported/mb2.pth')

# dummy_input = torch.randn((1, 3, 300, 300))
# f_path = 'exported/mobilenetv2.onnx'

# print('Exporting...')
# torch.onnx.export(model, dummy_input, f_path)
# print('Exported!')

# print('Simplifying...')
# onnx_model = onnx.load(f_path)
# onnx_model, check = onnxsim.simplify(onnx_model)
# onnx.save(onnx_model, f_path)
# print('Simplified!')

blobconverter.from_openvino('exported/openvino/mobilenetv2_test.xml', 'exported/openvino/mobilenetv2_test.bin')