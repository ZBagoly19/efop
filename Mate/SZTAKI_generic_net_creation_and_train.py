import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

NUM_OF_INPUT_DATA_TYPES = 100
NUM_OF_OUTPUT_DATA_TYPES = 10
EPOCHS = 1500

LearnRateSchedule = 'piecewise'
LearnRateDropPeriod = 150
LearnRateDropFactor = 0.7

glob_losses = []

INIT_LRS = [0.1]
#WEIGHT_DECAYS = [0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005]

shuffled_test_data = np.load("Mate_data_test_part__" + ".npy", allow_pickle=True)
shuffled_test_data = shuffled_test_data[:, 0:2]
shuffled_train_data = np.load("Mate_data_train_part__" + ".npy", allow_pickle=True)
shuffled_train_data = shuffled_train_data[:, 0:2]

NUMBERS_OF_LAYERS = [0]
WIDTHES_OF_LAYERS = [0]

for learn_rate in INIT_LRS:
    print("new initial learning rate:", learn_rate)
    # for bn in BATCH_NORMS:
        # for wd in WEIGHT_DECAYS:
            #print(wd, str(wd))
    for num_of_layers in NUMBERS_OF_LAYERS:
        # print(num_of_layers)
        for width in WIDTHES_OF_LAYERS:
            for k in [0]:
                #print("K =", k)
                #FILE_PRE = "tovabbtanulas_bemutatas_long_long_08_" + str(k) + "_" + str(learn_rate) + "_sched_300_085_bn_no_reg_no"

                # print(num_of_layers, width[width_idx])
                # print(width)
                # TIME = int(time.time())
                print("háló mélysége:", num_of_layers, "réteg szélesség:", width)
                #TRAINING_NAME = f"efop__{NUMBER_OF_LAYERS}"
                #log_name = TRAINING_NAME
                
                #print(TRAINING_NAME)
                device = torch.device("cpu")
                
                losses = np.zeros([EPOCHS, 3], dtype=float)
                
                # GPU
                def on_gpu():
                    if torch.cuda.is_available():
                        return torch.device("cuda:0")
                    else:
                        return torch.device("cpu")
                
                device = on_gpu()
                print(device)
                
                
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
                        
                        x = torch.randn(NUM_OF_INPUT_DATA_TYPES).view(-1, 1, NUM_OF_INPUT_DATA_TYPES)
                        self._to_linear = None
                        self.convs(x)
                        
                        self.fc1 = nn.Linear(self._to_linear, 512)
                        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
                        self.fc2 = nn.Linear(512, NUM_OF_OUTPUT_DATA_TYPES)
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

                    def __init__(self):
                        super().__init__()
                        
                        x = torch.randn(NUM_OF_INPUT_DATA_TYPES).view(-1, 1, NUM_OF_INPUT_DATA_TYPES)
                        self._to_linear = None
                        self.linearize(x)
                        
                        self.fc1 = nn.Linear(self._to_linear, NUM_OF_OUTPUT_DATA_TYPES)
                        #nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
                        if 1 < num_of_layers:
                            self.fc2 = nn.Linear(width, width)
                            #nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
                        if 2 < num_of_layers:
                            self.fc3 = nn.Linear(width, width)
                            #nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
                            self.fc4 = nn.Linear(width, width)
                            #nn.init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
                        if 4 < num_of_layers:
                            self.fc5 = nn.Linear(width, width)
                            #nn.init.kaiming_uniform_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
                            self.fc6 = nn.Linear(width, width)
                            #nn.init.kaiming_uniform_(self.fc6.weight, mode='fan_in', nonlinearity='relu')
                        if 6 < num_of_layers:
                            self.fc7 = nn.Linear(width, width)
                            #nn.init.kaiming_uniform_(self.fc7.weight, mode='fan_in', nonlinearity='relu')
                            self.fc8 = nn.Linear(width, width)
                            #nn.init.kaiming_uniform_(self.fc8.weight, mode='fan_in', nonlinearity='relu')
                        if 8 < num_of_layers:
                            self.fc9 = nn.Linear(width, width)
                            #nn.init.kaiming_uniform_(self.fc9.weight, mode='fan_in', nonlinearity='relu')
                            self.fc10 = nn.Linear(width, width)
                            #nn.init.kaiming_uniform_(self.fc10.weight, mode='fan_in', nonlinearity='relu')
                        if 10 < num_of_layers:
                            self.fc11 = nn.Linear(width, width)
                            #nn.init.kaiming_uniform_(self.fc11.weight, mode='fan_in', nonlinearity='relu')
                            self.fc12 = nn.Linear(width, width)
                            #nn.init.kaiming_uniform_(self.fc12.weight, mode='fan_in', nonlinearity='relu')
                        if 12 < num_of_layers:
                            self.fc13 = nn.Linear(width, width)
                            #nn.init.kaiming_uniform_(self.fc13.weight, mode='fan_in', nonlinearity='relu')
                            self.fc14 = nn.Linear(width, width)
                            #nn.init.kaiming_uniform_(self.fc14.weight, mode='fan_in', nonlinearity='relu')
                        #self.fc_last = nn.Linear(width, NUM_OF_OUTPUT_DATA_TYPES)
                        #nn.init.kaiming_uniform_(self.fc_last.weight, mode='fan_in', nonlinearity='relu')
                        
                    def linearize(self, x):
                        if self._to_linear == None:
                            self._to_linear = x[0].shape[0] * x[0].shape[1]
                
                        return x

                    def forward(self, x):
                        # print("33", x.shape, x)
                        
                        x = x.view(-1, self._to_linear)
                        #print("4", x.shape, x[0], x)
                        
                        x = self.fc1(x)
                        #x = F.relu(self.fc1(x))
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
                        
                        #x = self.fc_last(x)
                        #print("last", x.shape, x[0], x)
                        
                        return x
                

                net = Net_fc()
                net.to(device)
                optimizer = optim.Adam(net.parameters(), lr=learn_rate) # , weight_decay=wd)
                                       
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LearnRateDropPeriod, gamma=LearnRateDropFactor)

                
                
                loss_function = nn.MSELoss()
                
                def data_division_and_shaping():
                    #print([i[0] for i in shuffled_train_data])
                    #print(shuffled_train_data[1][1].shape)
                    my_train_X_l = torch.Tensor([i[0] for i in shuffled_train_data]).view(-1, NUM_OF_INPUT_DATA_TYPES)
                    my_test_X_l = torch.Tensor([i[0] for i in shuffled_test_data]).view(-1, NUM_OF_INPUT_DATA_TYPES)
                    # print("X", my_X.shape)
                    my_train_y_l = torch.Tensor([i[1] for i in shuffled_train_data]).view(-1, NUM_OF_OUTPUT_DATA_TYPES)
                    my_test_y_l = torch.Tensor([i[1] for i in shuffled_test_data]).view(-1, NUM_OF_OUTPUT_DATA_TYPES)
                    # print("y", my_y.shape)
                    
                    print("X test", my_test_X_l.shape)
                    print("Y test", my_test_y_l.shape)
                    print("X train", my_train_X_l.shape)
                    print("Y train", my_train_y_l.shape)
                    
                    return my_test_X_l, my_test_y_l, my_train_X_l, my_train_y_l
                
                my_test_X, my_test_y, my_train_X, my_train_y = data_division_and_shaping()
                                
                def my_test(size=len(my_test_X), print_now=False):
                    # print(my_test_X.shape, my_test_X)
                    # print("len(my_test_X)", len(my_test_X))
                    # print(size)
                    # random_start = np.random.randint(len(my_test_X) - size)
                    start = 0
                    
                    my_X2, my_y2 = my_test_X[start : start + size], my_test_y[start : start + size]
                    net.eval()
                    with torch.no_grad():
                        val_acc, val_loss = my_fwd_pass(my_X2.view(-1, 1, NUM_OF_INPUT_DATA_TYPES).to(device), my_y2.to(device), train=False)
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
                    #print("outputs:", outputs.shape)
                    accuracy = 0
                    loss = loss_function(outputs, b_y)
                    
                    if train:
                        loss.backward()
                        optimizer.step()
                
                    return accuracy, loss
                
                def my_train():
                    print("Training starts")
                    for epoch in range(EPOCHS):
                        lets_print = False
                        if epoch % 100 == 99:
                            lets_print = True
                            print("")
                            print("Epoch:", epoch)
                            print(optimizer)
                            
                        
                        my_batch_x = my_train_X[ : ].view(-1, 1, NUM_OF_INPUT_DATA_TYPES).to(device)
                        my_batch_y = my_train_y[ : ].to(device)
                        # print("batch_x", my_batch_x.shape)
                        # print("batch_y", my_batch_y.shape)
                        my_fwd_pass(my_batch_x, my_batch_y, train=True)

                        
                        accuracy_, loss_ = my_fwd_pass(my_batch_x, my_batch_y, train=True)
                        val_acc_, val_loss_ = my_test(print_now=lets_print)
                        a = loss_.cpu()
                        b = val_loss_.cpu()
                        c = a.detach().numpy()
                        d = b.detach().numpy()
                        losses_row = np.column_stack([epoch, c, d])
                        losses[epoch] = losses_row

                        if LearnRateSchedule == 'piecewise':
                            scheduler.step()
                        
                        if epoch == EPOCHS - 1:
                            path = os.path.join('Mate_halo__width_{}_depth_{}.npy'.format(width, num_of_layers))
                            print(path, epoch)
                            net.eval()
                            torch.save(net, path)
                            net.train()
                
                my_train()
                np.save('Mate_losses__width_{}_depth_{}.npy'.format(width, num_of_layers), losses)
                print("")
                print("rejtett rétegek száma:", num_of_layers, "rejtett rétegek szélessége:", width)
                glob_losses.append([learn_rate, False, 0, width, num_of_layers, sum(losses[-10 : , 2])/10])
                print("")
