"""
@author: Bagoly Zoltán
"""

import onnx

onnx_model = onnx.load("multiple_road_sensors_1500epoch_1_0.005_sched_400_08_bn_no_reg_no_net_DevAngsDistVel_WA___64_4_1399.onnx")
onnx_model = onnx.load("tinyyolov2-7.onnx")
onnx.checker.check_model(onnx_model)
onnx.helper.printable_graph(onnx_model.graph)
