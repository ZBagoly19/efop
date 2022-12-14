r% after running this, the workspace should be saved and it will hold the parameters 
% of the neural network, loadable into a net created in matlab, 
% usable with CarMaker

modelfile = "E:\Workspaces\.spyder-py3\csak_WheAng__speed50_cc0_acc333___0_0.001_sched_300_0.69_bn_no_reg_no_net_DevAngsDistVel_WA___64_4_1499.onnx";
params = importONNXFunction(modelfile, 'netFcn');

clear modelfile

fc1w = params.Learnables.fc1_weight;
fc1b = params.Learnables.fc1_bias;
fc2w = params.Learnables.fc2_weight;
fc2b = params.Learnables.fc2_bias;
fc3w = params.Learnables.fc3_weight;
fc3b = params.Learnables.fc3_bias;
fc4w = params.Learnables.fc4_weight;
fc4b = params.Learnables.fc4_bias;
fc_last_w = params.Learnables.fc_last_weight;
fc_last_b = params.Learnables.fc_last_bias;

clear params
save('SZTAKI_weights_and_biases')
