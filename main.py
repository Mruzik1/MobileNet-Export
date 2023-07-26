import sys
sys.path.append('./mobilenets-ssd-pytorch')

import torch
import onnx
import onnxsim
from custom_models import *


model = create_mobilenetv1_ssd_lite(80, is_test=True)
dummy_input = torch.randn((1, 3, 300, 300))
f_path = 'exported/mobilenet_nms.onnx'

# print('Exporting...')
# torch.onnx.export(model, dummy_input, f_path)
# print('Exported!')

# print('Simplifying...')
# onnx_model = onnx.load(f_path)
# onnx_model, check = onnxsim.simplify(onnx_model)
# onnx.save(onnx_model, f_path)
# print('Simplified!')