### set the parameters here ###
net_type = "convolutional"      # "fully connected" or "convolutional"
optim_type = "adam"             # "adam"

NUM_OF_INPUT_DATA_TYPES = 100
NUM_OF_OUTPUT_DATA_TYPES = 10
EPOCHS = 150
SAVE_EVERY_EPOCH = 100

LearnRateDropPeriod = 150
LearnRateDropFactor = 0.7

learn_rate = 0.1
# weight_decay = 0.00005

num_of_layers = 0
width = 10

FILE_PRE = 'Mate_'
###############################################################################♦
### no need to touch anything here ###
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if net_type == "convolutional":
    shuffled_test_data = np.load("Mate_data_test_2d__" + ".npy", allow_pickle=True)
    shuffled_test_data = shuffled_test_data[:, 0:2] # 3. oszlop: id, ha újra sorba rendeznénk
    shuffled_train_data = np.load("Mate_data_train_2d__" + ".npy", allow_pickle=True)
    shuffled_train_data = shuffled_train_data[:, 0:2] # 3. oszlop: id, ha újra sorba rendeznénk
elif net_type == "fully connected":
    shuffled_test_data = np.load("Mate_data_test_part__" + ".npy", allow_pickle=True)
    shuffled_test_data = shuffled_test_data[:, 0:2] # 3. oszlop: id, ha újra sorba rendeznénk
    shuffled_train_data = np.load("Mate_data_train_part__" + ".npy", allow_pickle=True)
    shuffled_train_data = shuffled_train_data[:, 0:2] # 3. oszlop: id, ha újra sorba rendeznénk
else:
    print("unknown net type")


 # GPU
def on_gpu():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

device = on_gpu()
print("device:", device)

# print("new initial learning rate:", learn_rate)
# print(wd, str(wd))
# print(num_of_layers)
# print("K =", k)
# print(num_of_layers, width)
# print("háló mélysége:", num_of_layers, "rétegek szélessége:", width)

device = torch.device("cpu")

losses = np.zeros([EPOCHS, 3], dtype=float)


x_c1, x_c2 = 0, 0
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

        x = torch.randn(10, 10).view(-1, 1, 10, 10)
        self._to_linear = None
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 32)  # self._to_linear = 40
        #nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        self.fc2 = nn.Linear(32, 16)
        #nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        self.fcLast = nn.Linear(16, NUM_OF_OUTPUT_DATA_TYPES)
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
        
        if self._to_linear == None:
            # print("\n", x_c1.shape)
            # print("\n", x_c2.shape)
            self._to_linear = x_c1.shape[2] * x_c1.shape[3]  +  x_c2.shape[2] * x_c2.shape[3]
            # print("_to_linear", self._to_linear)

        return x_c1, x_c2
        
    def forward(self, x):
        xc1, xc2 = self.convs(x)
        print("after convs", xc1.shape, xc2.shape)
        
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
        
        if num_of_layers != 0:
            self.fc1 = nn.Linear(self._to_linear, width)
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
        if num_of_layers != 0:
            self.fc_last = nn.Linear(width, NUM_OF_OUTPUT_DATA_TYPES)
            #nn.init.kaiming_uniform_(self.fc_last.weight, mode='fan_in', nonlinearity='relu')
        else:
            self.fc_last = nn.Linear(self._to_linear, NUM_OF_OUTPUT_DATA_TYPES)
            #nn.init.kaiming_uniform_(self.fc_last.weight, mode='fan_in', nonlinearity='relu')
        
    def linearize(self, x):
        if self._to_linear == None:
            self._to_linear = x[0].shape[0] * x[0].shape[1]

        return x

    def forward(self, x):
        # print("33", x.shape, x)
        
        x = x.view(-1, self._to_linear)
        #print("4", x.shape, x[0], x)
        
        if num_of_layers != 0:
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
        #print("last", x.shape, x[0], x)
        
        return x


if net_type == "fully connected":
    net = Net_fc()
elif net_type == "convolutional":
    net = Net_conv()
net.to(device)
if optim_type == "adam":
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
    
    # print("X test", my_test_X_l.shape)
    # print("Y test", my_test_y_l.shape)
    # print("X train", my_train_X_l.shape)
    # print("Y train", my_train_y_l.shape)
    
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
        print("Val loss: ", val_loss.item())
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

def train():
    print("Training starts")
    print("rejtett rétegek száma:", num_of_layers, "rejtett rétegek szélessége:", width)
    for epoch in range(EPOCHS):
        lets_print = False
        if epoch % 100 == 99:
            lets_print = True
            print("")
            print("Epoch:", epoch)
            # print(optimizer)
            
        
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

        scheduler.step()
        
        if epoch % SAVE_EVERY_EPOCH == SAVE_EVERY_EPOCH - 1:
            path = os.path.join(
                FILE_PRE + 'halo__width_{}_depth_{}_lr_{}_LearnRateDropPeriod_{}_LearnRateDropFactor_{}_epoch_{}.npy'.format(
                    width, num_of_layers, learn_rate, LearnRateDropPeriod, LearnRateDropFactor, epoch))
            print(path)
            print(net)
            net.eval()
            torch.save(net, path)
            net.train()
    np.save(FILE_PRE + 
        'losses__width_{}_depth_{}_lr_{}_LearnRateDropPeriod_{}_LearnRateDropFactor_{}.npy'.format(
            width, num_of_layers, learn_rate, LearnRateDropPeriod, LearnRateDropFactor), losses)

train()
