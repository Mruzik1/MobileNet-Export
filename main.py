import sys
sys.path.append('./mobilenets-ssd-pytorch')

import torch
import onnx
import onnxsim
from custom_models import *


model = create_mobilenetv1_ssd_lite(80)
dummy_input = torch.randn((1, 3, 640, 640))
f_path = 'exported/mobilenet_no_nms.onnx'

torch.onnx.export(model, dummy_input, f_path)

onnx_model = onnx.load(f_path)
onnx_model, check = onnxsim.simplify(onnx_model)
onnx.save(onnx_model, f_path)