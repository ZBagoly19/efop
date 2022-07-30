"""
@author: Bagoly Zoltán
"""

import os
import torch
import torch.nn as nn
import SZTAKI_constants as consts
import torch.nn.functional as F
import torch.onnx

batch_size = 1
k = 1
learn_rate = 0.005
FILE_PRE = "multiple_road_sensors_1500epoch_" + str(k) + "_" + str(learn_rate) + "_sched_400_08_bn_no_reg_no" + "_"
FILE_NAME_START = 'DevAngsDistVel_WA___'
NUM_OF_LAYERS = 4
WIDTH_OF_LAYERS = 64
EPOCH = 1399
#LEN_OF_INPUT = consts.NUM_OF_INPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS
LEN_OF_INPUT = consts.NUM_OF_INPUT_DATA_TYPES
#LEN_OF_OUTPUT = int(consts.NUM_OF_OUTPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS * consts.INPUT_OUTPUT_COEFFICIENT)
LEN_OF_OUTPUT = consts.NUM_OF_OUTPUT_DATA_TYPES
num_of_layers = NUM_OF_LAYERS
class Net_fc(nn.Module):
    def __init__(self):
        super().__init__()
        
        x = torch.randn(LEN_OF_INPUT).view(-1, 1, LEN_OF_INPUT)
        self._to_linear = None
        self.linearize(x)
        
        self.fc1 = nn.Linear(self._to_linear, WIDTH_OF_LAYERS)
        #nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        if 1 < num_of_layers:
            self.fc2 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            #nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        if 2 < num_of_layers:
            self.fc3 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            #nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
            self.fc4 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            #nn.init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
        if 4 < num_of_layers:
            self.fc5 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            #nn.init.kaiming_uniform_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
            self.fc6 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            #nn.init.kaiming_uniform_(self.fc6.weight, mode='fan_in', nonlinearity='relu')
        if 6 < num_of_layers:
            self.fc7 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            #nn.init.kaiming_uniform_(self.fc7.weight, mode='fan_in', nonlinearity='relu')
            self.fc8 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            #nn.init.kaiming_uniform_(self.fc8.weight, mode='fan_in', nonlinearity='relu')
        if 8 < num_of_layers:
            self.fc9 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            #nn.init.kaiming_uniform_(self.fc9.weight, mode='fan_in', nonlinearity='relu')
            self.fc10 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            #nn.init.kaiming_uniform_(self.fc10.weight, mode='fan_in', nonlinearity='relu')
        if 10 < num_of_layers:
            self.fc11 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            #nn.init.kaiming_uniform_(self.fc11.weight, mode='fan_in', nonlinearity='relu')
            self.fc12 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            #nn.init.kaiming_uniform_(self.fc12.weight, mode='fan_in', nonlinearity='relu')
        if 12 < num_of_layers:
            self.fc13 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            #nn.init.kaiming_uniform_(self.fc13.weight, mode='fan_in', nonlinearity='relu')
            self.fc14 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            #nn.init.kaiming_uniform_(self.fc14.weight, mode='fan_in', nonlinearity='relu')
        self.fc_last = nn.Linear(WIDTH_OF_LAYERS, LEN_OF_OUTPUT)
        #nn.init.kaiming_uniform_(self.fc_last.weight, mode='fan_in', nonlinearity='relu')
        
    def linearize(self, x):
        if self._to_linear == None:
            self._to_linear = x[0].shape[0] * x[0].shape[1]

        return x
        
    def forward(self, x):
        # print("33", x.shape, x[0], x)
        
        x = x.view(-1, self._to_linear)
        # print("4", x.shape, x[0], x)
        
        x = F.relu(self.fc1(x))
        # print("5", x.shape, x[0], x)
        
        if 1 < num_of_layers:
            x = F.relu(self.fc2(x))
            # print("6", x.shape, x[0], x)
        
        if 2 < num_of_layers:
            x = F.relu(self.fc3(x))
            #print("7", x.shape, x[0], x)
            x = F.relu(self.fc4(x))
            # print("8", x.shape, x[0], x)
            
        if 4 < num_of_layers:
            x = F.relu(self.fc5(x))
            #print("9", x.shape, x[0], x)
            x = F.relu(self.fc6(x))
            #print("10", x.shape, x[0], x)
            
        if 6 < num_of_layers:
            x = F.relu(self.fc7(x))
            #print("11", x.shape, x[0], x)
            x = F.relu(self.fc8(x))
            #print("12", x.shape, x[0], x)
            
        if 8 < num_of_layers:
            x = F.relu(self.fc9(x))
            #print("13", x.shape, x[0], x)
            x = F.relu(self.fc10(x))
            #print("14", x.shape, x[0], x)
            
        if 10 < num_of_layers:
            x = F.relu(self.fc11(x))
            #print("15", x.shape, x[0], x)
            x = F.relu(self.fc12(x))
            #print("16, x.shape, x[0], x)
        
        if 12 < num_of_layers:
            x = F.relu(self.fc13(x))
            #print("17", x.shape, x[0], x)
            x = F.relu(self.fc14(x))
            #print("18, x.shape, x[0], x)
        
        x = self.fc_last(x)
        # print("last", x.shape, x[0], x)
        
        return x
# GPU
def on_gpu():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")
DEVICE = on_gpu()
f = FILE_PRE + 'net_' + FILE_NAME_START + '{}_{}_{}.pth'.format(WIDTH_OF_LAYERS, NUM_OF_LAYERS, EPOCH)
print("próba:", f)
if os.path.isfile(f):
    print("valid filenév")
    net = torch.load(os.path.join(f))
    print("sikeres betöltés")
    net.eval()
    rand_input = torch.randn(batch_size, 1, 43, requires_grad=True).to(DEVICE)
    torch_out = net(rand_input)

    # Export the model
    torch.onnx.export(net,          # model being run
        rand_input,                 # model input (or a tuple for multiple inputs)
        FILE_PRE + 'net_' + FILE_NAME_START + '{}_{}_{}.onnx'.format(WIDTH_OF_LAYERS, NUM_OF_LAYERS, EPOCH),   
                                    # where to save the model (can be a file or file-like object)
        export_params=True,         # store the trained parameter weights inside the model file
        opset_version=9,            # the ONNX version to export the model to
        do_constant_folding=True,   # whether to execute constant folding for optimization
        input_names = ['input'],    # the model's input names
        output_names = ['output'],  # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                      'output' : {0 : 'batch_size'}})
