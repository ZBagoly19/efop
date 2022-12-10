"""
@author: Bagoly Zoltán
"""

import onnx

onnx_model = onnx.load("ipg_driver_cc1___0_0.005_sched_300_085_bn_no_reg_no_net_DevAngsDistVel_WA___64_4_1099.onnx")
onnx.checker.check_model(onnx_model)
onnx.helper.printable_graph(onnx_model.graph)
