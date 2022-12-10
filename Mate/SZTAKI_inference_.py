import os
import math
import numpy as np
### itt kell megadni a filenevet ###
PATH = 'fc2  2022-12-05 13-53-14'

##############################################################################
# ezt nem kell piszkalni

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

import SZTAKI_generic_net_creation_and_train as net_train

checkpoint = torch.load(os.path.join(PATH))
settings = net_train.settingsClass()
settings.net_type = checkpoint['net_type']
settings.net_name = checkpoint['net_name']
settings.optim_type = checkpoint['optim_type']
settings.NUM_OF_INPUT_DATA_TYPES_1st_dim = checkpoint['NUM_OF_INPUT_DATA_TYPES_1st_dim']
settings.NUM_OF_INPUT_DATA_TYPES_2st_dim = checkpoint['NUM_OF_INPUT_DATA_TYPES_2st_dim']
settings.NUM_OF_INPUT_DATA_TYPES = checkpoint['NUM_OF_INPUT_DATA_TYPES']
settings.NUM_OF_OUTPUT_DATA_TYPES = checkpoint['NUM_OF_OUTPUT_DATA_TYPES']
settings.train_vector = checkpoint['train_vector']
settings.test_vector = checkpoint['test_vector']
settings.EPOCHS = checkpoint['EPOCHS']
settings.SAVE_EVERY_EPOCH = checkpoint['SAVE_EVERY_EPOCH']
settings.LearnRateDropPeriod = checkpoint['LearnRateDropPeriod']
settings.LearnRateDropFactor = checkpoint['LearnRateDropFactor']
settings.learn_rate = checkpoint['learn_rate']
settings.weight_decay_ = checkpoint['weight_decay_']
settings.num_of_layers = checkpoint['num_of_layers']
settings.width = checkpoint['width']
settings.FILE_PRE = checkpoint['FILE_PRE']

# GPU
def on_gpu():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

colours = ['black', 'blue', 'green', 'brown', 'red', 'cyan', 'magenta', 
           'yellow', 'darkblue', 'orange', 'pink', 'beige', 'coral', 'crimson', 
           'darkgreen', 'fuchsia', 'goldenrod', 'grey', 'yellowgreen', 
           'lightblue', 'lime', 'navy', 'sienna', 'silver',
           'orchid', 'wheat', 'white', 'chocolate', 'khaki', 'azure',
           'salmon', 'plum']
styles = ['solid', 'dotted', 'dashed']
styles = ['-', '--', '-.', ':', 'solid']

DEVICE = on_gpu()

loss_function = nn.MSELoss()

# Convolutional nets
class Net_conv1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 1))#, stride=4)
        #nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(3, 1))#, dilation=1, padding=2)
        #nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.conv3 = nn.Conv2d(1, 1, kernel_size=(3, 1))#, dilation=1, padding=2)
        #nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        self.conv4 = nn.Conv2d(1, 1, kernel_size=(3, 1))#, dilation=1, padding=2)
        #nn.init.kaiming_uniform_(self.conv4.weight, mode='fan_in', nonlinearity='relu')
        
        self.conv5 = nn.Conv2d(1, 1, kernel_size=(1, 3))#, dilation=1, padding=2)
        #nn.init.kaiming_uniform_(self.conv5.weight, mode='fan_in', nonlinearity='relu')
        self.conv6 = nn.Conv2d(1, 1, kernel_size=(1, 3))#, dilation=1, padding=2)
        #nn.init.kaiming_uniform_(self.conv6.weight, mode='fan_in', nonlinearity='relu')
        self.conv7 = nn.Conv2d(1, 1, kernel_size=(1, 3))#, dilation=1, padding=2)
        #nn.init.kaiming_uniform_(self.conv7.weight, mode='fan_in', nonlinearity='relu')
        self.conv8 = nn.Conv2d(1, 1, kernel_size=(1, 3))#, dilation=1, padding=2)
        #nn.init.kaiming_uniform_(self.conv7.weight, mode='fan_in', nonlinearity='relu')

        x = torch.randn(settings.NUM_OF_INPUT_DATA_TYPES_1st_dim, settings.NUM_OF_INPUT_DATA_TYPES_2st_dim).view(
            -1, 1, settings.NUM_OF_INPUT_DATA_TYPES_1st_dim, settings.NUM_OF_INPUT_DATA_TYPES_2st_dim)
        self._to_linear_k1 = None
        self._to_linear_k2 = None
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear_k1 + self._to_linear_k2, 32)  # self._to_linear_k1 = 20
        #nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        self.fc2 = nn.Linear(32, 16)
        #nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        self.fcLast = nn.Linear(16, settings.NUM_OF_OUTPUT_DATA_TYPES)
        #nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def convs(self, x):

        x_c1 = F.relu(self.conv1(x))
        # print("1", x_c1[0].shape)
        x_c1 = F.relu(self.conv2(x_c1))
        # print("2", x_c1[0].shape)
        x_c1 = F.relu(self.conv3(x_c1))
        # print("3", x_c1[0].shape)
        x_c1 = F.relu(self.conv4(x_c1))
        # print("4", x_c1[0].shape)
        # print("\n")
        
        x_c2 = F.relu(self.conv5(x))
        # print("5", x_c2[0].shape)
        x_c2 = F.relu(self.conv6(x_c2))
        # print("6", x_c2[0].shape)
        x_c2 = F.relu(self.conv7(x_c2))
        # print("7", x_c2[0].shape)
        x_c2 = F.relu(self.conv8(x_c2))
        # print("8", x_c2[0].shape)
        # print("\n")
        
        # print("x", x[0].shape)
        
        #x = F.max_pool1d(F.relu(self.conv1(x)), (3))
        
        if self._to_linear_k1 == None:
            # print("\n", x_c1.shape)
            self._to_linear_k1 = x_c1.shape[2] * x_c1.shape[3]
            #print("_to_linear_k1", self._to_linear_k1)
        if self._to_linear_k2 == None:
            # print("\n", x_c2.shape)
            self._to_linear_k2 = x_c2.shape[2] * x_c2.shape[3]
            #print("_to_linear_k2", self._to_linear_k2)

        return x_c1, x_c2
        
    def forward(self, x):
        xc1, xc2 = self.convs(x)
        #print("after convs", xc1.shape, xc2.shape)
        
        xc1_1d = xc1.view(-1, self._to_linear_k1)
        xc2_1d = xc2.view(-1, self._to_linear_k2)
        #print("xc1_1d", xc1_1d.size())
        #print("xc2_1d", xc2_1d.size())
        xc_concat = torch.cat((xc1_1d, xc2_1d), 1) # 1: dim to concatenate on, data length
        #print("xc_concat", xc_concat.size())
        
        out = F.relu(self.fc1(xc_concat))
        #print("fc1", out.shape)
                
        out = F.relu(self.fc2(out))
        #print("fc2", out.shape)
        
        out = F.relu(self.fcLast(out))
        #print("fc_last", out.shape)
        
        return out

# Fully connected nets
class Net_fc1(nn.Module):
        
    def __init__(self):
        super().__init__()
        
        x = torch.randn(settings.NUM_OF_INPUT_DATA_TYPES).view(-1, 1, settings.NUM_OF_INPUT_DATA_TYPES)
        self._to_linear = None
        self.linearize(x)
        
        if settings.num_of_layers != 0:
            self.fc1 = nn.Linear(self._to_linear, settings.width)
            #nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        if 1 < settings.num_of_layers:
            self.fc2 = nn.Linear(settings.width, settings.width)
            #nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        if 2 < settings.num_of_layers:
            self.fc3 = nn.Linear(settings.width, settings.width)
            #nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
            self.fc4 = nn.Linear(settings.width, settings.width)
            #nn.init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
        if 4 < settings.num_of_layers:
            self.fc5 = nn.Linear(settings.width, settings.width)
            #nn.init.kaiming_uniform_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
            self.fc6 = nn.Linear(settings.width, settings.width)
            #nn.init.kaiming_uniform_(self.fc6.weight, mode='fan_in', nonlinearity='relu')
        if 6 < settings.num_of_layers:
            self.fc7 = nn.Linear(settings.width, settings.width)
            #nn.init.kaiming_uniform_(self.fc7.weight, mode='fan_in', nonlinearity='relu')
            self.fc8 = nn.Linear(settings.width, settings.width)
            #nn.init.kaiming_uniform_(self.fc8.weight, mode='fan_in', nonlinearity='relu')
        if 8 < settings.num_of_layers:
            self.fc9 = nn.Linear(settings.width, settings.width)
            #nn.init.kaiming_uniform_(self.fc9.weight, mode='fan_in', nonlinearity='relu')
            self.fc10 = nn.Linear(settings.width, settings.width)
            #nn.init.kaiming_uniform_(self.fc10.weight, mode='fan_in', nonlinearity='relu')
        if 10 < settings.num_of_layers:
            self.fc11 = nn.Linear(settings.width, settings.width)
            #nn.init.kaiming_uniform_(self.fc11.weight, mode='fan_in', nonlinearity='relu')
            self.fc12 = nn.Linear(settings.width, settings.width)
            #nn.init.kaiming_uniform_(self.fc12.weight, mode='fan_in', nonlinearity='relu')
        if 12 < settings.num_of_layers:
            self.fc13 = nn.Linear(settings.width, settings.width)
            #nn.init.kaiming_uniform_(self.fc13.weight, mode='fan_in', nonlinearity='relu')
            self.fc14 = nn.Linear(settings.width, settings.width)
            #nn.init.kaiming_uniform_(self.fc14.weight, mode='fan_in', nonlinearity='relu')
        if settings.num_of_layers != 0:
            self.fc_last = nn.Linear(settings.width, settings.NUM_OF_OUTPUT_DATA_TYPES)
            #nn.init.kaiming_uniform_(self.fc_last.weight, mode='fan_in', nonlinearity='relu')
        else:
            self.fc_last = nn.Linear(self._to_linear, settings.NUM_OF_OUTPUT_DATA_TYPES)
            #nn.init.kaiming_uniform_(self.fc_last.weight, mode='fan_in', nonlinearity='relu')
        
    def linearize(self, x):
        if self._to_linear == None:
            self._to_linear = x[0].shape[0] * x[0].shape[1]

        return x

    def forward(self, x):
        # print("33", x.shape, x)
        
        x = x.view(-1, self._to_linear)
        #print("4", x.shape, x[0], x)
        
        if settings.num_of_layers != 0:
            x = F.relu(self.fc1(x))
            # print("5", x.shape, x[0], x)
        
        if 1 < settings.num_of_layers:
            x = F.relu(self.fc2(x))
            # print("6", x.shape, x[0], x)
        
        if 2 < settings.num_of_layers:
            x = F.relu(self.fc3(x))
            #print("7", x.shape, x[0], x)
            x = F.relu(self.fc4(x))
            # print("8", x.shape, x[0], x)
            
        if 4 < settings.num_of_layers:
            x = F.relu(self.fc5(x))
            #print("9", x.shape, x[0], x)
            x = F.relu(self.fc6(x))
            #print("10", x.shape, x[0], x)
            
        if 6 < settings.num_of_layers:
            x = F.relu(self.fc7(x))
            #print("11", x.shape, x[0], x)
            x = F.relu(self.fc8(x))
            #print("12", x.shape, x[0], x)
            
        if 8 < settings.num_of_layers:
            x = F.relu(self.fc9(x))
            #print("13", x.shape, x[0], x)
            x = F.relu(self.fc10(x))
            #print("14", x.shape, x[0], x)
            
        if 10 < settings.num_of_layers:
            x = F.relu(self.fc11(x))
            #print("15", x.shape, x[0], x)
            x = F.relu(self.fc12(x))
            #print("16, x.shape, x[0], x)
        
        if 12 < settings.num_of_layers:
            x = F.relu(self.fc13(x))
            #print("17", x.shape, x[0], x)
            x = F.relu(self.fc14(x))
            #print("18, x.shape, x[0], x)
        
        x = self.fc_last(x)
        #print("last", x.shape, x[0], x)
        
        return x

class Net_fc2(nn.Module):
        
    def __init__(self):
        super().__init__()
        
        x = torch.randn(settings.NUM_OF_INPUT_DATA_TYPES).view(-1, 1, settings.NUM_OF_INPUT_DATA_TYPES)
        self._to_linear = None
        self.linearize(x)
        
        self.fc1 = nn.Linear(self._to_linear, settings.width)
        #nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        self.fc2 = nn.Linear(settings.width, settings.width)
        #nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        self.fc3 = nn.Linear(settings.width, settings.width)
        #nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        self.fc4 = nn.Linear(settings.width, settings.width)
        #nn.init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
        self.fc5 = nn.Linear(settings.width, settings.width)
        #nn.init.kaiming_uniform_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
        self.fc6 = nn.Linear(settings.width, settings.width)
        #nn.init.kaiming_uniform_(self.fc6.weight, mode='fan_in', nonlinearity='relu')
        self.fc7 = nn.Linear(settings.width, settings.width)
        #nn.init.kaiming_uniform_(self.fc7.weight, mode='fan_in', nonlinearity='relu')
        self.fc8 = nn.Linear(settings.width, settings.width)
        #nn.init.kaiming_uniform_(self.fc8.weight, mode='fan_in', nonlinearity='relu')
        
        self.fc_last = nn.Linear(settings.width, settings.NUM_OF_OUTPUT_DATA_TYPES)
        #nn.init.kaiming_uniform_(self.fc_last.weight, mode='fan_in', nonlinearity='relu')
        
        
    def linearize(self, x):
        if self._to_linear == None:
            self._to_linear = x[0].shape[0] * x[0].shape[1]
    
        return x
    
    def forward(self, x):
        # print("33", x.shape, x)
        
        x = x.view(-1, self._to_linear)
        #print("4", x.shape, x[0], x)
        
        x = F.relu(self.fc1(x))
        # print("5", x.shape, x[0], x)
    
        x = F.relu(self.fc2(x))
        # print("6", x.shape, x[0], x)
    
        x = F.relu(self.fc3(x))
        #print("7", x.shape, x[0], x)
        x = F.relu(self.fc4(x))
        # print("8", x.shape, x[0], x)
        
        x = F.relu(self.fc5(x))
        #print("9", x.shape, x[0], x)
        x = F.relu(self.fc6(x))
        #print("10", x.shape, x[0], x)
        
        x = F.relu(self.fc7(x))
        #print("11", x.shape, x[0], x)
        x = F.relu(self.fc8(x))
        #print("12", x.shape, x[0], x)
            
        
        
        x = self.fc_last(x)
        #print("last", x.shape, x[0], x)
        
        return x


parsed_path = PATH.split("  ")
if parsed_path[0] == "fc1":
    net = Net_fc1()
if parsed_path[0] == "fc2":
    net = Net_fc2()
if parsed_path[0] == "cnn1":
    net = Net_conv1()()
net.load_state_dict(checkpoint['state_dict'])
net.eval()


