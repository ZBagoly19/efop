###############################################################################♦
### no need to touch anything here ###
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import SZTAKI_constants_ as consts

class trainClass():

    def data_division_and_shaping(self):
        #print([i[0] for i in shuffled_train_data])
        #print(shuffled_train_data[1][1].shape)
        if consts.net_type == "convolutional":
            my_train_X_l = torch.Tensor([i[0] for i in self.shuffled_train_data]).view(
                -1, consts.NUM_OF_INPUT_DATA_TYPES_1st_dim, consts.NUM_OF_INPUT_DATA_TYPES_2st_dim)
            my_test_X_l = torch.Tensor([i[0] for i in self.shuffled_test_data]).view(
                -1, consts.NUM_OF_INPUT_DATA_TYPES_1st_dim, consts.NUM_OF_INPUT_DATA_TYPES_2st_dim)
        elif consts.net_type == "fully connected":
            my_train_X_l = torch.Tensor([i[0] for i in self.shuffled_train_data]).view(-1, consts.NUM_OF_INPUT_DATA_TYPES)
            my_test_X_l = torch.Tensor([i[0] for i in self.shuffled_test_data]).view(-1, consts.NUM_OF_INPUT_DATA_TYPES)
        else:
            print("unknown net type:", consts.net_type)
        
        # print("X", my_X.shape)
        my_train_y_l = torch.Tensor([i[1] for i in self.shuffled_train_data]).view(-1, consts.NUM_OF_OUTPUT_DATA_TYPES)
        my_test_y_l = torch.Tensor([i[1] for i in self.shuffled_test_data]).view(-1, consts.NUM_OF_OUTPUT_DATA_TYPES)
        # print("y", my_y.shape)
        
        # print("X test", my_test_X_l.shape)
        # print("Y test", my_test_y_l.shape)
        # print("X train", my_train_X_l.shape)
        # print("Y train", my_train_y_l.shape)
        
        self.my_test_X = my_test_X_l
        self.my_test_y = my_test_y_l
        self.my_train_X = my_train_X_l
        self.my_train_y = my_train_y_l
        
        #return my_test_X_l, my_test_y_l, my_train_X_l, my_train_y_l
        
    # GPU
    def on_gpu(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")
    
    
    #print("device:", self.device)


    def setup(self):

        if consts.net_type == "convolutional":
            self.shuffled_test_data = np.load("Mate_data_test_2d__" + ".npy", allow_pickle=True)
            self.shuffled_test_data = self.shuffled_test_data[:, 0:2] # 3. oszlop: id, ha újra sorba rendeznénk
            self.shuffled_train_data = np.load("Mate_data_train_2d__" + ".npy", allow_pickle=True)
            self.shuffled_train_data = self.shuffled_train_data[:, 0:2] # 3. oszlop: id, ha újra sorba rendeznénk
        elif consts.net_type == "fully connected":
            self.shuffled_test_data = np.load("Mate_data_test_part__" + ".npy", allow_pickle=True)
            self.shuffled_test_data = self.shuffled_test_data[:, 0:2] # 3. oszlop: id, ha újra sorba rendeznénk
            self.shuffled_train_data = np.load("Mate_data_train_part__" + ".npy", allow_pickle=True)
            self.shuffled_train_data = self.shuffled_train_data[:, 0:2] # 3. oszlop: id, ha újra sorba rendeznénk
        else:
            print("unknown net type:", consts.net_type)
        
        
        
        # print("new initial learning rate:", learn_rate)
        # print(wd, str(wd))
        # print(num_of_layers)
        # print("K =", k)
        # print(num_of_layers, width)
        # print("háló mélysége:", num_of_layers, "rétegek szélessége:", width)
                
        self.losses = np.zeros([consts.EPOCHS, 3], dtype=float)
        print(self.losses.shape)
        
        
        # Net convolutional
        class Net_conv(nn.Module):
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
        
                x = torch.randn(consts.NUM_OF_INPUT_DATA_TYPES_1st_dim, consts.NUM_OF_INPUT_DATA_TYPES_2st_dim).view(
                    -1, 1, consts.NUM_OF_INPUT_DATA_TYPES_1st_dim, consts.NUM_OF_INPUT_DATA_TYPES_2st_dim)
                self._to_linear_k1 = None
                self._to_linear_k2 = None
                self.convs(x)
                
                self.fc1 = nn.Linear(self._to_linear_k1 + self._to_linear_k2, 32)  # self._to_linear_k1 = 20
                #nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
                self.fc2 = nn.Linear(32, 16)
                #nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
                self.fcLast = nn.Linear(16, consts.NUM_OF_OUTPUT_DATA_TYPES)
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
        
        # Net fully connected
        class Net_fc(nn.Module):
        
            def __init__(self):
                super().__init__()
                
                x = torch.randn(consts.NUM_OF_INPUT_DATA_TYPES).view(-1, 1, consts.NUM_OF_INPUT_DATA_TYPES)
                self._to_linear = None
                self.linearize(x)
                
                if consts.num_of_layers != 0:
                    self.fc1 = nn.Linear(self._to_linear, consts.width)
                    #nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
                if 1 < consts.num_of_layers:
                    self.fc2 = nn.Linear(consts.width, consts.width)
                    #nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
                if 2 < consts.num_of_layers:
                    self.fc3 = nn.Linear(consts.width, consts.width)
                    #nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
                    self.fc4 = nn.Linear(consts.width, consts.width)
                    #nn.init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
                if 4 < consts.num_of_layers:
                    self.fc5 = nn.Linear(consts.width, consts.width)
                    #nn.init.kaiming_uniform_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
                    self.fc6 = nn.Linear(consts.width, consts.width)
                    #nn.init.kaiming_uniform_(self.fc6.weight, mode='fan_in', nonlinearity='relu')
                if 6 < consts.num_of_layers:
                    self.fc7 = nn.Linear(consts.width, consts.width)
                    #nn.init.kaiming_uniform_(self.fc7.weight, mode='fan_in', nonlinearity='relu')
                    self.fc8 = nn.Linear(consts.width, consts.width)
                    #nn.init.kaiming_uniform_(self.fc8.weight, mode='fan_in', nonlinearity='relu')
                if 8 < consts.num_of_layers:
                    self.fc9 = nn.Linear(consts.width, consts.width)
                    #nn.init.kaiming_uniform_(self.fc9.weight, mode='fan_in', nonlinearity='relu')
                    self.fc10 = nn.Linear(consts.width, consts.width)
                    #nn.init.kaiming_uniform_(self.fc10.weight, mode='fan_in', nonlinearity='relu')
                if 10 < consts.num_of_layers:
                    self.fc11 = nn.Linear(consts.width, consts.width)
                    #nn.init.kaiming_uniform_(self.fc11.weight, mode='fan_in', nonlinearity='relu')
                    self.fc12 = nn.Linear(consts.width, consts.width)
                    #nn.init.kaiming_uniform_(self.fc12.weight, mode='fan_in', nonlinearity='relu')
                if 12 < consts.num_of_layers:
                    self.fc13 = nn.Linear(consts.width, consts.width)
                    #nn.init.kaiming_uniform_(self.fc13.weight, mode='fan_in', nonlinearity='relu')
                    self.fc14 = nn.Linear(consts.width, consts.width)
                    #nn.init.kaiming_uniform_(self.fc14.weight, mode='fan_in', nonlinearity='relu')
                if consts.num_of_layers != 0:
                    self.fc_last = nn.Linear(consts.width, consts.NUM_OF_OUTPUT_DATA_TYPES)
                    #nn.init.kaiming_uniform_(self.fc_last.weight, mode='fan_in', nonlinearity='relu')
                else:
                    self.fc_last = nn.Linear(self._to_linear, consts.NUM_OF_OUTPUT_DATA_TYPES)
                    #nn.init.kaiming_uniform_(self.fc_last.weight, mode='fan_in', nonlinearity='relu')
                
            def linearize(self, x):
                if self._to_linear == None:
                    self._to_linear = x[0].shape[0] * x[0].shape[1]
        
                return x
        
            def forward(self, x):
                # print("33", x.shape, x)
                
                x = x.view(-1, self._to_linear)
                #print("4", x.shape, x[0], x)
                
                if consts.num_of_layers != 0:
                    x = F.relu(self.fc1(x))
                    # print("5", x.shape, x[0], x)
                
                if 1 < consts.num_of_layers:
                    x = F.relu(self.fc2(x))
                    # print("6", x.shape, x[0], x)
                
                if 2 < consts.num_of_layers:
                    x = F.relu(self.fc3(x))
                    #print("7", x.shape, x[0], x)
                    x = F.relu(self.fc4(x))
                    # print("8", x.shape, x[0], x)
                    
                if 4 < consts.num_of_layers:
                    x = F.relu(self.fc5(x))
                    #print("9", x.shape, x[0], x)
                    x = F.relu(self.fc6(x))
                    #print("10", x.shape, x[0], x)
                    
                if 6 < consts.num_of_layers:
                    x = F.relu(self.fc7(x))
                    #print("11", x.shape, x[0], x)
                    x = F.relu(self.fc8(x))
                    #print("12", x.shape, x[0], x)
                    
                if 8 < consts.num_of_layers:
                    x = F.relu(self.fc9(x))
                    #print("13", x.shape, x[0], x)
                    x = F.relu(self.fc10(x))
                    #print("14", x.shape, x[0], x)
                    
                if 10 < consts.num_of_layers:
                    x = F.relu(self.fc11(x))
                    #print("15", x.shape, x[0], x)
                    x = F.relu(self.fc12(x))
                    #print("16, x.shape, x[0], x)
                
                if 12 < consts.num_of_layers:
                    x = F.relu(self.fc13(x))
                    #print("17", x.shape, x[0], x)
                    x = F.relu(self.fc14(x))
                    #print("18, x.shape, x[0], x)
                
                x = self.fc_last(x)
                #print("last", x.shape, x[0], x)
                
                return x
        
        
        self.device = self.on_gpu()
        
        if consts.net_type == "fully connected":
            self.net = Net_fc()
        elif consts.net_type == "convolutional":
            self.net = Net_conv()
            #print("created")
        self.net.to(self.device)
        if consts.optim_type == "adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=consts.learn_rate) # , weight_decay=wd)
                               
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=consts.LearnRateDropPeriod, gamma=consts.LearnRateDropFactor)
        
        
        
        self.loss_function = nn.MSELoss()
        
        self.data_division_and_shaping()
        
    
    
    def my_test(self, size=1, print_now=False):
        # print(my_test_X.shape, my_test_X)
        # print("len(my_test_X)", len(my_test_X))
        # print(size)
        # random_start = np.random.randint(len(my_test_X) - size)
        start = 0
        
        my_X2, my_y2 = self.my_test_X[start : start + size], self.my_test_y[start : start + size]
        self.net.eval()
        with torch.no_grad():
            if consts.net_type == "convolutional":
                val_acc, val_loss = self.my_fwd_pass(my_X2.view(
                    #-1, 1, consts.NUM_OF_INPUT_DATA_TYPES_1st_dim, consts.NUM_OF_INPUT_DATA_TYPES_2st_dim), my_y2, train=False)
                    -1, 1, consts.NUM_OF_INPUT_DATA_TYPES_1st_dim, consts.NUM_OF_INPUT_DATA_TYPES_2st_dim).to(self.device), my_y2.to(self.device), train=False)
            elif consts.net_type == "fully connected":
                #val_acc, val_loss = self.my_fwd_pass(my_X2.view(-1, 1, consts.NUM_OF_INPUT_DATA_TYPES), my_y2, train=False)
                val_acc, val_loss = self.my_fwd_pass(my_X2.view(-1, 1, consts.NUM_OF_INPUT_DATA_TYPES).to(self.device), my_y2.to(self.device), train=False)
            else:
                print("unknown net type:", consts.net_type)
        if(print_now):
            print("Val loss: ", val_loss.item())
        self.net.train()
        return val_acc, val_loss
    
    def my_fwd_pass(self, b_x, b_y, train=False):
        if train == True:
            self.net.zero_grad()
        #print("b_x", b_x.shape)
        #print("b_y:", b_y.shape)
        outputs = self.net(b_x)
        #print("outputs:", outputs.shape)
        accuracy = 0
        loss = self.loss_function(outputs, b_y)
        
        if train:
            loss.backward()
            self.optimizer.step()
    
        return accuracy, loss

    def train(self):
        nets = []
        
        print("Training starts")
        #print("rejtett rétegek száma:", num_of_layers, "rejtett rétegek szélessége:", width)
        for epoch in tqdm.tqdm(range(consts.EPOCHS)):
            lets_print = False
            if epoch % 100 == 99:
                lets_print = True
                print("")
                print("Epoch:", epoch)
                # print(optimizer)
                
            if consts.net_type == "convolutional":
                my_batch_x = self.my_train_X[ : ].view(
                    -1, 1, consts.NUM_OF_INPUT_DATA_TYPES_1st_dim, consts.NUM_OF_INPUT_DATA_TYPES_2st_dim).to(self.device)
            elif consts.net_type == "fully connected":
                my_batch_x = self.my_train_X[ : ].view(-1, 1, consts.NUM_OF_INPUT_DATA_TYPES).to(self.device)
            else:
                print("unknown net type:", consts.net_type)
            my_batch_y = self.my_train_y[ : ].to(self.device)
            # print("batch_x", my_batch_x.shape)
            # print("batch_y", my_batch_y.shape)
            self.my_fwd_pass(my_batch_x, my_batch_y, train=True)
    
            
            accuracy_, loss_ = self.my_fwd_pass(my_batch_x, my_batch_y, train=True)
            val_acc_, val_loss_ = self.my_test(size=len(self.my_test_X), print_now=lets_print)
            a = loss_.cpu()
            b = val_loss_.cpu()
            c = a.detach().numpy()
            d = b.detach().numpy()
            losses_row = np.column_stack([epoch, c, d])
            self.losses[epoch] = losses_row
    
            self.scheduler.step()
            
            if epoch % consts.SAVE_EVERY_EPOCH == consts.SAVE_EVERY_EPOCH - 1:
                path = os.path.join(
                    consts.FILE_PRE + 'halo__width_{}_depth_{}_lr_{}_LearnRateDropPeriod_{}_LearnRateDropFactor_{}_epoch_{}.npy'.format(
                        consts.width, consts.num_of_layers, consts.learn_rate, consts.LearnRateDropPeriod, consts.LearnRateDropFactor, epoch))
                print(path)
                print(self.net)
                print('\n')
                self.net.eval()
                # torch.save(self.net, path) # ez igy error, nem enged local (self) valtozot pickle-ezni
                nets.append(self.net)
                self.net.train()
        np.save(consts.FILE_PRE + 
            'losses__width_{}_depth_{}_lr_{}_LearnRateDropPeriod_{}_LearnRateDropFactor_{}.npy'.format(
                consts.width, consts.num_of_layers, consts.learn_rate, consts.LearnRateDropPeriod, consts.LearnRateDropFactor), self.losses)
        return nets
    #train()
