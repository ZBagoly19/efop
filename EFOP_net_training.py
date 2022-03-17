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
import time

NET_TYPE = "fully connected"     # "convolutional" or "fully connected"
MODEL_NUMBER = 1
TIME_ = int(time.time())
print("time:", TIME_)
TRAINING_NAME = f"efop__{TIME_}"
log_name = TRAINING_NAME

print(TRAINING_NAME)
device = torch.device("cpu")

# GPU
def on_gpu():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

device = on_gpu()
print(device)

# adat makrok
LEN_OF_SEGMENTS = 400
LEN_OF_INPUT = 4 * LEN_OF_SEGMENTS
LEN_OF_OUTPUT = 5 * LEN_OF_SEGMENTS
# mat = scipy.io.loadmat("meas01.mat")

shuffled_testing_data = np.load("my_testing_data_1d_XYVO_better_skale.npy", allow_pickle=True)
shuffled_training_data = np.load("my_training_data_1d_XYVO_better_skale.npy", allow_pickle=True)
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
    def __init__(self):
        super().__init__()
        
        x = torch.randn(LEN_OF_INPUT).view(-1, 1, LEN_OF_INPUT)
        self._to_linear = None
        self.linearize(x)
        
        self.fc1 = nn.Linear(self._to_linear, 512)
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        self.fc2 = nn.Linear(512, 1024)
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        self.fc3 = nn.Linear(1024, 1024)
        nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        self.fc4 = nn.Linear(1024, 2048)
        nn.init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
        self.fc5 = nn.Linear(2048, 1024)
        nn.init.kaiming_uniform_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
        self.fc6 = nn.Linear(1024, 512)
        nn.init.kaiming_uniform_(self.fc6.weight, mode='fan_in', nonlinearity='relu')
        self.fc_last = nn.Linear(512, LEN_OF_OUTPUT)
        nn.init.kaiming_uniform_(self.fc_last.weight, mode='fan_in', nonlinearity='relu')
                
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
        
        x = F.relu(self.fc2(x))
        # print("6", x.shape, x[0], x)
        
        x = F.relu(self.fc3(x))
        # print("7", x.shape, x[0], x)
        
        x = F.relu(self.fc4(x))
        # print("8", x.shape, x[0], x)
        
        x = F.relu(self.fc5(x))
        # print("9", x.shape, x[0], x)
        
        x = F.relu(self.fc6(x))
        # print("10", x.shape, x[0], x)
        
        x = self.fc_last(x)
        # print("last", x.shape, x[0], x)
        
        return x


if NET_TYPE == "convolutional":
    net = Net_conv()
    optimizer = optim.Adam(net.parameters(), lr=0.001 #, weight_decay=0.00001)
                           )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.1)
if NET_TYPE == "fully connected":
    net = Net_fc()
    optimizer = optim.Adam(net.parameters(), lr=0.0002 #, weight_decay=0.00001)
                           )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=110, gamma=0.5)
else:
    print("Biztos jó a háló típus?")
    net = torch.load(os.path.join('net_164727849401148.pth'))
    optimizer = optim.Adam(net.parameters(), lr=0.0002 #, weight_decay=0.00001)
                           )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=110, gamma=0.5)

net.to(device)
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

def my_test(size=32, print_now=False):
    # print(my_test_X.shape, my_test_X)
    # print("len(my_test_X)", len(my_test_X))
    # print(size)
    random_start = np.random.randint(len(my_test_X) - size)
    
    my_X2, my_y2 = my_test_X[random_start : random_start + size], my_test_y[random_start : random_start + size]
    net.eval()
    with torch.no_grad():
        val_acc, val_loss = my_fwd_pass(my_X2.view(-1, 1, LEN_OF_INPUT).to(device), my_y2.to(device), train=False)
    if(print_now):
        print("Val loss: ", val_loss, "; Val_acc: ", val_acc)
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
    matches = [abs(torch.argmax(i) - torch.argmax(j)) < 0.1 for i, j in zip(outputs, b_y)]
    accuracy = matches.count(True) / len(matches)
    loss = loss_function(outputs, b_y)
    
    if train:
        loss.backward()
        optimizer.step()

    # with open("efop_log_output_of_net.log", "a") as file:
    #     # for i in range(len(outputs)):
    #     #     output = outputs[0][i]
    #     file.write(f"{TRAINING_NAME},{round(time.time(),3)},{round(float(outputs),10)}\n")
    #     file.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(loss),15)}\n")
    return accuracy, loss

def my_train():
    BATCH_SIZE = 16
    EPOCHS = 250
    cnt = 2
    print("Training starts")
    with open("efop_log.log", "a") as file:
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch)
            for i in my_tqdm.tqdm(range(0, len(my_train_X), BATCH_SIZE)):
                my_batch_x = my_train_X[i : i + BATCH_SIZE].view(-1, 1, LEN_OF_INPUT).to(device)
                my_batch_y = my_train_y[i : i + BATCH_SIZE].to(device)
                # print("batch_x", my_batch_x.shape, my_batch_x)
                # print("batch_y", my_batch_y)
                # print("train_x", my_train_X.shape)
                # print("train_y", my_train_y.shape)
                accuracy_, loss_ = my_fwd_pass(my_batch_x, my_batch_y, train=True)
                
            accuracy_, loss_ = my_fwd_pass(my_batch_x, my_batch_y, train=True)
            val_acc_, val_loss_ = my_test()
            file.write(f"{log_name},{cnt},{round(float(accuracy_),5)},{round(float(loss_),10)},{round(float(val_acc_),5)},{round(float(val_loss_),10)}\n")
            cnt += 1
            my_test(print_now=True)
            scheduler.step()
            path = os.path.join('net_{}.pth'.format(TIME_ *100000 + MODEL_NUMBER *1000 + epoch))
            print(path)
            torch.save(net, path)

# Tanitas nelkuli halo valasza
with open("efop_log.log", "a") as file:
    # file.write(f"{log_name},0,{round(float(3200),5)},3200,{round(float(3200),5)},3200\n")
    val_acc, val_loss = my_test()
    print("Tanitas nelkuli halo valaszan a kulonbseg:", val_loss)
    file.write(f"{log_name},1,{round(float(val_acc),5)},{round(float(val_loss),10)},{round(float(val_acc),5)},{round(float(val_loss),10)}\n")

my_train()
print(TIME_)
