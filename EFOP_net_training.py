"""
@author: Bagoly Zoltán
         zoltan.bagoly@gmail.com
"""
import os
import numpy as np
import tqdm as my_tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import time
import SZTAKI_constants as consts
# import random

randomise = False
NET_TYPE = "fully connected"     # "convolutional" or "fully connected"
glob_losses = []
#WIDTH_OF_LAYERS = 32
#NUMBER_OF_LAYERS = 12
#BATCH_NORMS = [False, True]
INIT_LRS = [0.001, 0.0007]
#WEIGHT_DECAYS = [0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005]

NUMBERS_OF_LAYERS = [1, 2, 3, 4, 5]
WIDTHES_OF_LAYERS = [32, 64]

for learn_rate in INIT_LRS:
    print("new initial learning rate:", learn_rate)
    # for bn in BATCH_NORMS:
        # for wd in WEIGHT_DECAYS:
            #print(wd, str(wd))
    for num_of_layers in NUMBERS_OF_LAYERS:
        # print(num_of_layers)
        for width in WIDTHES_OF_LAYERS:
            for k in range(5):
                print("K =", k)
                FILE_PRE = "multiple_road_sensors_" + str(k) + "_" + str(learn_rate) + "_sched_400_08_bn_no_reg_no"
                #if bn:
                    #FILE_PRE = "10_Adam_" + str(learn_rate) + "_sched_50_0.8_bn_no" + "_reg_no"
                # print(num_of_layers, WIDTHES_OF_LAYERS[width_idx])
                WIDTH_OF_LAYERS = width
                # print(WIDTH_OF_LAYERS)
                # TIME = int(time.time())
                print("rétegek:", num_of_layers, "réteg szélesség:", WIDTH_OF_LAYERS)
                #TRAINING_NAME = f"efop__{NUMBER_OF_LAYERS}"
                #log_name = TRAINING_NAME
                
                #print(TRAINING_NAME)
                device = torch.device("cpu")
                
                # adat makrok
                #LEN_OF_INPUT = consts.NUM_OF_INPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS
                LEN_OF_INPUT = consts.NUM_OF_INPUT_DATA_TYPES
                #LEN_OF_OUTPUT = int(consts.NUM_OF_OUTPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS * consts.INPUT_OUTPUT_COEFFICIENT)
                LEN_OF_OUTPUT = consts.NUM_OF_OUTPUT_DATA_TYPES
                # print(LEN_OF_OUTPUT)
                # mat = scipy.io.loadmat("meas01.mat")
                losses = np.zeros([consts.EPOCHS, 3], dtype=float)
                
                # GPU
                def on_gpu():
                    if torch.cuda.is_available():
                        return torch.device("cuda:0")
                    else:
                        return torch.device("cpu")
                
                device = on_gpu()
                print(device)
                
                shuffled_testing_data = np.load("my_testing_data_DevAngsDevDistVel_WheAng_1_k" + str(k) + ".npy", allow_pickle=True)
                shuffled_training_data = np.load("my_training_data_DevAngsDevDistVel_WheAng_1_k" + str(k) + ".npy", allow_pickle=True)
                # print("shu_td", shuffled_training_data.shape)
                # print("shu_td 0", shuffled_training_data[0].shape)
                # print("shu_td 1", shuffled_training_data[1].shape)
                # print("shu_td 00", shuffled_training_data[0][0].shape)
                # print("shu_td 01", shuffled_training_data[0][1].shape)
                # print("shu_td 10", shuffled_training_data[1][0].shape)
                # print("shu_td 11", shuffled_training_data[1][1].shape)
                
                
                # Net convolutional
                class Net_conv(nn.Module):
                    def __init__(self):
                        super().__init__()
                        
                        self.conv1 = nn.Conv1d(1, 32, kernel_size=4, stride=4)
                        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
                        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, dilation=1, padding=2)
                        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
                        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, dilation=1, padding=2)
                        nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
                        self.conv4 = nn.Conv1d(64, 64, kernel_size=5, dilation=1, padding=2)
                        nn.init.kaiming_uniform_(self.conv4.weight, mode='fan_in', nonlinearity='relu')
                        self.conv5 = nn.Conv1d(64, 64, kernel_size=5, dilation=1, padding=2)
                        nn.init.kaiming_uniform_(self.conv5.weight, mode='fan_in', nonlinearity='relu')
                        
                        self.conv6 = nn.Conv1d(16, 128, kernel_size=5, dilation=1, padding=2)
                        nn.init.kaiming_uniform_(self.conv6.weight, mode='fan_in', nonlinearity='relu')
                        self.conv7 = nn.Conv1d(128, 64, kernel_size=5, dilation=1, padding=2)
                        nn.init.kaiming_uniform_(self.conv7.weight, mode='fan_in', nonlinearity='relu')
                        
                        x = torch.randn(LEN_OF_INPUT).view(-1, 1, LEN_OF_INPUT)
                        self._to_linear = None
                        self.convs(x)
                        
                        self.fc1 = nn.Linear(self._to_linear, 512)
                        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
                        self.fc2 = nn.Linear(512, LEN_OF_OUTPUT)
                        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
                        
                    def convs(self, x):
                        x = F.max_pool1d(F.relu(self.conv1(x)), (3))
                        #print("\n")
                        #print("1", x[0].shape, x[0], x)
                        x = F.max_pool1d(F.relu(self.conv2(x)), (3))
                        #print("2", x[0].shape, x[0], x)
                        x = F.max_pool1d(F.relu(self.conv3(x)), (3))
                        #x = F.max_pool1d(F.relu(self.conv4(x)), (3))
                        #x = F.max_pool1d(F.relu(self.conv5(x)), (3))
                        #x = F.max_pool1d(F.relu(self.conv6(x)), (3))
                        #x = F.max_pool1d(F.relu(self.conv7(x)), (3))
                        # hat ezt kiszamolni nem lett volna konnyu. Ezert kell kiprobalni random adattal
                        
                        if self._to_linear == None:
                            self._to_linear = x[0].shape[0] * x[0].shape[1]
                
                        return x
                        
                    def forward(self, x):
                        # print(self.only_linear)
                        x = self.convs(x)
                        # print("3", x.shape, x[0], x)
                        
                        x = x.view(-1, self._to_linear)
                        # print("4", x.shape, x[0], x)
                        
                        x = F.relu(self.fc1(x))
                        # print("5", x.shape, x[0], x)
                                
                        x = self.fc2(x)
                        # print("6", x.shape, x[0], x)
                        
                        return x
                
                # Net fully connected
                class Net_fc(nn.Module):
                    """if bn:
                        def __init__(self):
                            super().__init__()
                            
                            x = torch.randn(LEN_OF_INPUT).view(-1, 1, LEN_OF_INPUT)
                            self._to_linear = None
                            self.linearize(x)
                            
                            self.fc1 = nn.Linear(self._to_linear, WIDTH_OF_LAYERS)
                            #nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
                            self.bn1 = nn.BatchNorm1d(WIDTH_OF_LAYERS, affine=True, track_running_stats=False, momentum=0.01)
                            self.fc2 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                            #nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
                            self.bn2 = nn.BatchNorm1d(WIDTH_OF_LAYERS, affine=False, track_running_stats=False, momentum=0.01)
                            if 2 < num_of_layers:
                                self.fc3 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                                #nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
                                self.bn3 = nn.BatchNorm1d(WIDTH_OF_LAYERS, affine=False, track_running_stats=False, momentum=0.01)
                                self.fc4 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                                #nn.init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
                                self.bn4 = nn.BatchNorm1d(WIDTH_OF_LAYERS, affine=False, track_running_stats=False, momentum=0.01)
                            if 4 < num_of_layers:
                                self.fc5 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                                #nn.init.kaiming_uniform_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
                                self.bn5 = nn.BatchNorm1d(WIDTH_OF_LAYERS, affine=False, track_running_stats=False, momentum=0.01)
                                self.fc6 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                                #nn.init.kaiming_uniform_(self.fc6.weight, mode='fan_in', nonlinearity='relu')
                                self.bn6 = nn.BatchNorm1d(WIDTH_OF_LAYERS, affine=False, track_running_stats=False, momentum=0.01)
                            if 6 < num_of_layers:
                                self.fc7 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                                #nn.init.kaiming_uniform_(self.fc7.weight, mode='fan_in', nonlinearity='relu')
                                self.bn7 = nn.BatchNorm1d(WIDTH_OF_LAYERS, affine=False, track_running_stats=False, momentum=0.01)
                                self.fc8 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                                #nn.init.kaiming_uniform_(self.fc8.weight, mode='fan_in', nonlinearity='relu')
                                self.bn8 = nn.BatchNorm1d(WIDTH_OF_LAYERS, affine=False, track_running_stats=False, momentum=0.01)
                            if 8 < num_of_layers:
                                self.fc9 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                                #nn.init.kaiming_uniform_(self.fc9.weight, mode='fan_in', nonlinearity='relu')
                                self.bn9 = nn.BatchNorm1d(WIDTH_OF_LAYERS, affine=False, track_running_stats=False, momentum=0.01)
                                self.fc10 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                                #nn.init.kaiming_uniform_(self.fc10.weight, mode='fan_in', nonlinearity='relu')
                                self.bn10 = nn.BatchNorm1d(WIDTH_OF_LAYERS, affine=False, track_running_stats=False, momentum=0.01)
                            if 10 < num_of_layers:
                                self.fc11 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                                #nn.init.kaiming_uniform_(self.fc11.weight, mode='fan_in', nonlinearity='relu')
                                self.bn11 = nn.BatchNorm1d(WIDTH_OF_LAYERS, affine=False, track_running_stats=False, momentum=0.01)
                                self.fc12 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                                #nn.init.kaiming_uniform_(self.fc12.weight, mode='fan_in', nonlinearity='relu')
                                self.bn12 = nn.BatchNorm1d(WIDTH_OF_LAYERS, affine=False, track_running_stats=False, momentum=0.01)
                            if 12 < num_of_layers:
                                self.fc13 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                                #nn.init.kaiming_uniform_(self.fc13.weight, mode='fan_in', nonlinearity='relu')
                                self.bn13 = nn.BatchNorm1d(WIDTH_OF_LAYERS, affine=False, track_running_stats=False, momentum=0.01)
                                self.fc14 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                                #nn.init.kaiming_uniform_(self.fc14.weight, mode='fan_in', nonlinearity='relu')
                                self.bn14 = nn.BatchNorm1d(WIDTH_OF_LAYERS, affine=False, track_running_stats=False, momentum=0.01)
                            self.fc_last = nn.Linear(WIDTH_OF_LAYERS, LEN_OF_OUTPUT)
                            #nn.init.kaiming_uniform_(self.fc_last.weight, mode='fan_in', nonlinearity='relu')
                    else:"""
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
                       
                    """if bn:
                        def forward(self, x):
                            # print("33", x.shape, x[0], x)
                            
                            x = x.view(-1, self._to_linear)
                            # print("4", x.shape, x[0], x)
                            
                            x = F.relu(self.bn1(self.fc1(x)))
                            # print("5", x.shape, x[0], x)
                            x = F.relu(self.bn2(self.fc2(x)))
                            # print("6", x.shape, x[0], x)
                            
                            if 2 < num_of_layers:
                                x = F.relu(self.bn3(self.fc3(x)))
                                #print("7", x.shape, x[0], x)
                                x = F.relu(self.bn4(self.fc4(x)))
                                # print("8", x.shape, x[0], x)
                                
                            if 4 < num_of_layers:
                                x = F.relu(self.bn5(self.fc5(x)))
                                #print("9", x.shape, x[0], x)
                                x = F.relu(self.bn6(self.fc6(x)))
                                #print("10", x.shape, x[0], x)
                                
                            if 6 < num_of_layers:
                                x = F.relu(self.bn7(self.fc7(x)))
                                #print("11", x.shape, x[0], x)
                                x = F.relu(self.bn8(self.fc8(x)))
                                #print("12", x.shape, x[0], x)
                                
                            if 8 < num_of_layers:
                                x = F.relu(self.bn9(self.fc9(x)))
                                #print("13", x.shape, x[0], x)
                                x = F.relu(self.bn10(self.fc10(x)))
                                #print("14", x.shape, x[0], x)
                                
                            if 10 < num_of_layers:
                                x = F.relu(self.bn11(self.fc11(x)))
                                #print("15", x.shape, x[0], x)
                                x = F.relu(self.bn12(self.fc12(x)))
                                #print("16, x.shape, x[0], x)
                            
                            if 12 < num_of_layers:
                                x = F.relu(self.bn13(self.fc13(x)))
                                #print("17", x.shape, x[0], x)
                                x = F.relu(self.bn14(self.fc14(x)))
                                #print("18, x.shape, x[0], x)
                            
                            x = self.fc_last(x)
                            # print("last", x.shape, x[0], x)
                            
                            return x
                    
                    else:"""
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
                
                
                # if NET_TYPE == "convolutional":
                #     net = Net_conv()
                #     optimizer = optim.Adam(net.parameters(), lr=0.001 #, weight_decay=0.00001)
                #                            )
                #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.5)
                if NET_TYPE == "fully connected":
                    net = Net_fc()
                    net.to(device)
                    optimizer = optim.Adam(net.parameters(), lr=learn_rate) # , weight_decay=wd)
                                           
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.8)
                # else:
                #     print("Háló betöltés!?")
                #     net = torch.load(os.path.join('net_164799577101180.pth'))
                #     optimizer = optim.Adam(net.parameters(), lr=0.0002 #, weight_decay=0.00001)
                #                            )
                #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=110, gamma=0.5)
                
                
                loss_function = nn.MSELoss()
                
                def data_division_and_shaping():
                    #print([i[0] for i in shuffled_training_data])
                    #print(shuffled_training_data[1][1].shape)
                    my_X = torch.Tensor([i[0] for i in shuffled_training_data]).view(-1, LEN_OF_INPUT)
                    my_X_t = torch.Tensor([i[0] for i in shuffled_testing_data]).view(-1, LEN_OF_INPUT)
                    # print("X", my_X.shape)
                    my_y = torch.Tensor([i[1] for i in shuffled_training_data]).view(-1, LEN_OF_OUTPUT)
                    my_y_t = torch.Tensor([i[1] for i in shuffled_testing_data]).view(-1, LEN_OF_OUTPUT)
                    # print("y", my_y.shape)
                    
                    # needed if the data is not devided already
                    #VAL_PCT = 0.1
                    #val_size = int(len(my_X) * VAL_PCT)
                    
                    #my_test_X = my_X[-val_size :]
                    #my_test_y = my_y[-val_size :]
                    #my_train_X = my_X[ : -val_size]
                    #my_train_y = my_y[ : -val_size]
                    
                    # needed if data is devided in data preparation (to shuffled_training_data and shuffled_testing_data)
                    my_test_X_l = my_X_t
                    my_test_y_l = my_y_t
                    my_train_X_l = my_X
                    my_train_y_l = my_y
                    
                    print("X test", my_test_X_l.shape)
                    print("Y test", my_test_y_l.shape)
                    print("X train", my_train_X_l.shape)
                    print("Y train", my_train_y_l.shape)
                    
                    return my_test_X_l, my_test_y_l, my_train_X_l, my_train_y_l
                
                my_test_X, my_test_y, my_train_X, my_train_y = data_division_and_shaping()
                
                # matrices = np.ones([consts.EPOCHS, len(my_test_X), 2, consts.NUM_OF_OUTPUT_DATA_TYPES, consts.LEN_OF_SEGMENTS], dtype=float)
                
                def my_test(size=len(my_test_X), print_now=False):
                    # print(my_test_X.shape, my_test_X)
                    # print("len(my_test_X)", len(my_test_X))
                    # print(size)
                    # random_start = np.random.randint(len(my_test_X) - size)
                    start = 0
                    
                    my_X2, my_y2 = my_test_X[start : start + size], my_test_y[start : start + size]
                    net.eval()
                    with torch.no_grad():
                        val_acc, val_loss = my_fwd_pass(my_X2.view(-1, 1, LEN_OF_INPUT).to(device), my_y2.to(device), train=False)
                    if(print_now):
                        print("Val loss: ", val_loss)
                    net.train()
                    return val_acc, val_loss
                
                def my_fwd_pass(b_x, b_y, train=False):
                    if train == True:
                        net.zero_grad()
                    #print("b_x", b_x.shape)
                    #print("b_y:", b_y.shape)
                    outputs = net(b_x)
                    #print("outputs: ", outputs.shape, outputs)
                    # if train == False:
                        # print("outputs:", outputs, "\nb_y:", b_y)
                    # a matches-t maskepp kene szamolni
                    # matches = [abs(torch.argmax(i) - torch.argmax(j)) < 0.1 for i, j in zip(outputs, b_y)]
                    accuracy = 0
                    # accuracy = matches.count(True) / len(matches)
                    loss = loss_function(outputs, b_y)
                    
                    if train:
                        loss.backward()
                        #torch.nn.utils.clip_grad_norm_(net.parameters(), 0.7)
                        optimizer.step()
                
                    # with open("efop_log_output_of_net.log", "a") as file:
                    #     # for i in range(len(outputs)):
                    #     #     output = outputs[0][i]
                    #     file.write(f"{TRAINING_NAME},{round(time.time(),3)},{round(float(outputs),10)}\n")
                    #     file.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(loss),15)}\n")
                    return accuracy, loss
                
                def one_segment_test(start, print_loss=False):
                    # print(start)
                    my_X3, my_y3 = my_test_X[start : start + 1], my_test_y[start : start + 1]
                    to_show_wanted = my_y3.to(device)
                    to_show_guessed = net(my_X3.view(-1, 1, LEN_OF_INPUT).to(device))
                    # print(to_show_wanted.shape)
                    # print(to_show_guessed.shape)
                    # print(" Target:", to_show_wanted, "\n", "Guess:", to_show_guessed)
                    loss = loss_function(to_show_guessed, to_show_wanted)
                    if print_loss:
                        print(loss)
                    return to_show_wanted, to_show_guessed
                def one_segment_test_on_train(start):
                    # print(start)
                    my_X3, my_y3 = my_train_X[start : start + 1], my_train_y[start : start + 1]
                    to_show_wanted = my_y3.to(device)
                    to_show_guessed = net(my_X3.view(-1, 1, LEN_OF_INPUT).to(device))
                    # print(to_show_wanted.shape)
                    # print(to_show_guessed.shape)
                    # print(" Target:", to_show_wanted, "\n", "Guess:", to_show_guessed)
                    return to_show_wanted, to_show_guessed
                def create_matrices(train_as_well=False):
                    matrix_test = np.zeros([len(my_train_X), LEN_OF_OUTPUT * 2], dtype=float)
                    for seg in range(0, len(my_test_X), 1):
                        wanted, guessed = one_segment_test(seg)
                        a = wanted.cpu()
                        b = guessed.cpu()
                        c = a.detach().numpy()
                        d = b.detach().numpy()
                        matrix_row = np.column_stack([c, d])
                        matrix_test[seg] = matrix_row
                
                    matrix_train = np.zeros([len(my_train_X), LEN_OF_OUTPUT * 2], dtype=float)
                    if train_as_well:
                        for seg in my_tqdm.tqdm(range(0, len(my_train_X), 1)):
                            wanted, guessed = one_segment_test_on_train(seg)
                            a = wanted.cpu()
                            b = guessed.cpu()
                            c = a.detach().numpy()
                            d = b.detach().numpy()
                            matrix_row = np.column_stack([c, d])
                            matrix_train[seg] = matrix_row
                    
                    return matrix_test, matrix_train
                
                def my_train():
                    print("Training starts")
                    # with open("efop_log.log", "a") as file:
                    for epoch in range(consts.EPOCHS):
                        lets_print = False
                        if epoch % 100 == 99:
                            lets_print = True
                            # print("")
                            # print(optimizer)
                            # print("Epoch:", epoch)
                        # cnt = 0
                        
                        #for b in range(int(len(my_train_X) / BATCH_SIZE)):
                        my_batch_x = my_train_X[ : ].view(-1, 1, LEN_OF_INPUT).to(device)
                        my_batch_y = my_train_y[ : ].to(device)
                        # print("batch_x", my_batch_x.shape)
                        # print("batch_y", my_batch_y.shape)
                        my_fwd_pass(my_batch_x, my_batch_y, train=True)
                        # cnt += 1
                        # print(cnt)
                        
                        # if randomise == False:
                        #     for i in range(0, len(my_train_X), BATCH_SIZE):
                        #         my_batch_x = my_train_X[i : i + BATCH_SIZE].view(-1, 1, LEN_OF_INPUT).to(device)
                        #         my_batch_y = my_train_y[i : i + BATCH_SIZE].to(device)
                        #         # print("batch_x", my_batch_x.shape)
                        #         # print("batch_y", my_batch_y.shape)
                        #         # print("train_x", my_train_X.shape)
                        #         # print("train_y", my_train_y.shape)
                        #         my_fwd_pass(my_batch_x, my_batch_y, train=True)
                        
                        # else:
                        #     for i in range(0, len(my_train_X), BATCH_SIZE):
                        #         segs_x = [None] * BATCH_SIZE
                        #         segs_y = [None] * BATCH_SIZE
                        #         for j in range(BATCH_SIZE):
                        #             rand = random.randint(0, len(my_train_X) - 1)
                        #             segs_x[j] = my_train_X[rand].view(1, 1, LEN_OF_INPUT)
                        #             segs_y[j] = my_train_y[rand].view(1, LEN_OF_OUTPUT)
                        #             # print(segs_y[j].shape)
                        #         my_batch_x = torch.cat((segs_x)).to(device)
                        #         my_batch_y = torch.cat((segs_y)).to(device)
                        #         # print("batch_x", my_batch_x.shape)
                        #         # print("batch_y", my_batch_y.shape)
                        #         my_fwd_pass(my_batch_x, my_batch_y, train=True)
                        
                        accuracy_, loss_ = my_fwd_pass(my_batch_x, my_batch_y, train=True)
                        val_acc_, val_loss_ = my_test(print_now=lets_print)
                        a = loss_.cpu()
                        b = val_loss_.cpu()
                        c = a.detach().numpy()
                        d = b.detach().numpy()
                        losses_row = np.column_stack([epoch, c, d])
                        losses[epoch] = losses_row
                        # file.write(f"{log_name},{cnt},{round(float(accuracy_),5)},{round(float(loss_),10)},{round(float(val_acc_),5)},{round(float(val_loss_),10)}\n")
                        # cnt += 1
                        #if 90 < epoch:
                        scheduler.step()
                        #print(epoch, optimizer)
                        if 250 < epoch:
                            if epoch % consts.EPOCH_SAVE_CONST == consts.EPOCH_SAVE_CONST - 1:
                                path = os.path.join(FILE_PRE + '_net_DevAngsDistVel_WA___{}_{}_{}.pth'.format(WIDTH_OF_LAYERS, num_of_layers, epoch))
                                print(path)
                                torch.save(net, path)
                        #if lets_print:
                            #print(net)
                            #print(optimizer)
                        # matrix_test_, matrix_train_ = create_matrices()
                        # matrix_test_g_formed = np.zeros([len(my_test_X), 2, consts.NUM_OF_OUTPUT_DATA_TYPES, consts.LEN_OF_SEGMENTS], dtype=float)
                        # for test_case in range(len(my_test_X)):
                        #     for j in range(LEN_OF_OUTPUT):
                        #         if j % consts.NUM_OF_OUTPUT_DATA_TYPES == 0:
                        #             matrix_test_g_formed[test_case][0][0][int(j / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_test_[test_case][j]
                        #         if j % consts.NUM_OF_OUTPUT_DATA_TYPES == 1:
                        #             matrix_test_g_formed[test_case][0][1][int(j / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_test_[test_case][j]
                        #         # if j % consts.NUM_OF_OUTPUT_DATA_TYPES == 2:
                        #         #     matrix_test_g_formed[test_case][0][2][int(j / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_test_[test_case][j]
                        #         # if j % consts.NUM_OF_OUTPUT_DATA_TYPES == 3:
                        #         #     matrix_test_g_formed[test_case][0][3][int(j / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_test_[test_case][j]
                        #         # if j % consts.NUM_OF_OUTPUT_DATA_TYPES == 4:
                        #         #     matrix_test_g_formed[test_case][0][4][int(j / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_test_[test_case][j]
                        #     for k in range(LEN_OF_OUTPUT, LEN_OF_OUTPUT * 2):
                        #         if k % consts.NUM_OF_OUTPUT_DATA_TYPES == 0:
                        #             matrix_test_g_formed[test_case][1][0][int((k-LEN_OF_OUTPUT) / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_test_[test_case][k]
                        #         if k % consts.NUM_OF_OUTPUT_DATA_TYPES == 1:
                        #             matrix_test_g_formed[test_case][1][1][int((k-LEN_OF_OUTPUT) / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_test_[test_case][k]
                        #         if k % consts.NUM_OF_OUTPUT_DATA_TYPES == 2:
                        #             matrix_test_g_formed[test_case][1][2][int((k-LEN_OF_OUTPUT) / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_test_[test_case][k]
                        #         if k % consts.NUM_OF_OUTPUT_DATA_TYPES == 3:
                        #             matrix_test_g_formed[test_case][1][3][int((k-LEN_OF_OUTPUT) / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_test_[test_case][k]
                        #         if k % consts.NUM_OF_OUTPUT_DATA_TYPES == 4:
                        #             matrix_test_g_formed[test_case][1][4][int((k-LEN_OF_OUTPUT) / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_test_[test_case][k]
                        # matrices[epoch] = matrix_test_g_formed
                        
                
                # Tanitas nelkuli halo valasza
                # with open("efop_log.log", "a") as file:
                #     # file.write(f"{log_name},0,{round(float(3200),5)},3200,{round(float(3200),5)},3200\n")
                #     val_acc, val_loss = my_test()
                #     print("Tanitas nelkuli halo valaszan a kulonbseg:", val_loss)
                #     file.write(f"{log_name},1,{round(float(val_acc),5)},{round(float(val_loss),10)},{round(float(val_acc),5)},{round(float(val_loss),10)}\n")
                
                my_train()
                np.save(FILE_PRE + '_DevAngsDistVel_WA___losses_and_val_losses_of_{}_{}.npy'.format(WIDTH_OF_LAYERS, num_of_layers), losses)
                #np.save('XYO_VAV__matrices_of_{}_{}.npy'.format(WIDTH_OF_LAYERS, NUMBER_OF_LAYERS), matrices)
                print("")
                print("rejtett rétegek száma:", num_of_layers, "rejtett rétegek szélessége:", WIDTH_OF_LAYERS)
                glob_losses.append([learn_rate, False, 0, width, num_of_layers, sum(losses[-10 : , 2])/10])
                #print(glob_losses)
                print("")
#np.save("glob_losses_np.npy", glob_losses)!
