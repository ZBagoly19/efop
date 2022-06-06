"""
@author: Bagoly Zoltán
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import torch
import torch.nn as nn
import torch.nn.functional as F
import SZTAKI_constants as consts
style.use("ggplot")
loss_function = torch.nn.MSELoss()

##############################################################################
# itt állítjuk be, mi érdekel
    # ezekkel adjuk meg, mely fájt töltjük be
file_name_start = 'XYO_VAV__'
WIDTH_OF_LAYERS = 16
NUM_OF_LAYERS = 2
EPOCH = 399
    # ezzel pedig, hogy mely szegmenseket rajzolja
test_segments = np.array([0, 5])    # matrix_test_glob-ból
train_segments = np.array([0])          # matrix_train_glob-ból
figure_const = 1

##############################################################################
# ehhez nem kell nyúlni
if consts.EPOCHS < EPOCH:
    print("Túl magas a epoch érték!", EPOCH)

# GPU
def on_gpu():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

DEVICE = on_gpu()

loss_function = nn.MSELoss()

LEN_OF_INPUT = consts.NUM_OF_INPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS
LEN_OF_OUTPUT = consts.NUM_OF_OUTPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS

SHUF_TEST_DATA = np.load("my_testing_data_XYO_VAV_better_skale.npy", allow_pickle=True)
SHUF_TRAIN_DATA = np.load("my_training_data_XYO_VAV_better_skale.npy", allow_pickle=True)
TRAIN_X = torch.Tensor([i[0] for i in SHUF_TRAIN_DATA]).view(-1, consts.NUM_OF_INPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS)
TEST_X = torch.Tensor([i[0] for i in SHUF_TEST_DATA]).view(-1, consts.NUM_OF_INPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS)
TRAIN_Y = torch.Tensor([i[1] for i in SHUF_TRAIN_DATA]).view(-1, consts.NUM_OF_OUTPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS)
TEST_Y = torch.Tensor([i[1] for i in SHUF_TEST_DATA]).view(-1, consts.NUM_OF_OUTPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS)

matrix_test_glob = np.zeros([len(TEST_X), 2, consts.NUM_OF_OUTPUT_DATA_TYPES, consts.LEN_OF_SEGMENTS], dtype=float)
matrix_train_glob = np.zeros([len(TRAIN_X), 2, consts.NUM_OF_OUTPUT_DATA_TYPES, consts.LEN_OF_SEGMENTS], dtype=float)

# Net fully connected
class Net_fc(nn.Module):
    def __init__(self):
        super().__init__()
        
        x = torch.randn(LEN_OF_INPUT).view(-1, 1, LEN_OF_INPUT)
        self._to_linear = None
        self.linearize(x)
        
        self.fc1 = nn.Linear(self._to_linear, WIDTH_OF_LAYERS)
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        self.fc2 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        if 2 < NUM_OF_LAYERS:
            self.fc3 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
            self.fc4 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            nn.init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
        if 4 < NUM_OF_LAYERS:
            self.fc5 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            nn.init.kaiming_uniform_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
            self.fc6 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            nn.init.kaiming_uniform_(self.fc6.weight, mode='fan_in', nonlinearity='relu')
        if 6 < NUM_OF_LAYERS:
            self.fc7 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            nn.init.kaiming_uniform_(self.fc7.weight, mode='fan_in', nonlinearity='relu')
            self.fc8 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            nn.init.kaiming_uniform_(self.fc8.weight, mode='fan_in', nonlinearity='relu')
        if 8 < NUM_OF_LAYERS:
            self.fc9 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            nn.init.kaiming_uniform_(self.fc9.weight, mode='fan_in', nonlinearity='relu')
            self.fc10 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            nn.init.kaiming_uniform_(self.fc10.weight, mode='fan_in', nonlinearity='relu')
        if 10 < NUM_OF_LAYERS:
            self.fc11 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            nn.init.kaiming_uniform_(self.fc11.weight, mode='fan_in', nonlinearity='relu')
            self.fc12 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            nn.init.kaiming_uniform_(self.fc12.weight, mode='fan_in', nonlinearity='relu')
        if 12 < NUM_OF_LAYERS:
            self.fc13 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            nn.init.kaiming_uniform_(self.fc13.weight, mode='fan_in', nonlinearity='relu')
            self.fc14 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
            nn.init.kaiming_uniform_(self.fc14.weight, mode='fan_in', nonlinearity='relu')
        self.fc_last = nn.Linear(WIDTH_OF_LAYERS, LEN_OF_OUTPUT)
        nn.init.kaiming_uniform_(self.fc_last.weight, mode='fan_in', nonlinearity='relu')
    def linearize(self, x):
        if self._to_linear == None:
            self._to_linear = x[0].shape[0] * x[0].shape[1]
        return x
        
    def forward(self, x):        
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if 2 < NUM_OF_LAYERS:
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
        if 4 < NUM_OF_LAYERS:
            x = F.relu(self.fc5(x))
            x = F.relu(self.fc6(x))
        if 6 < NUM_OF_LAYERS:
            x = F.relu(self.fc7(x))
            x = F.relu(self.fc8(x))
        if 8 < NUM_OF_LAYERS:
            x = F.relu(self.fc9(x))
            x = F.relu(self.fc10(x))
        if 10 < NUM_OF_LAYERS:
            x = F.relu(self.fc11(x))
            x = F.relu(self.fc12(x))
        if 12 < NUM_OF_LAYERS:
            x = F.relu(self.fc13(x))
            x = F.relu(self.fc14(x))
        x = self.fc_last(x)        
        return x

f = 'net_XYO_VAV__{}_{}_{}.pth'.format(WIDTH_OF_LAYERS, NUM_OF_LAYERS, EPOCH)
print("próba:", f)
if os.path.isfile(f):
    print("betöltött háló", f)
    
    net = torch.load(os.path.join(f))
    net.eval()
    
    def my_test(size=len(TEST_X), print_now=False):
        start = 0
        my_X2, my_y2 = TEST_X[start : start + size], TEST_Y[start : start + size]
        net.eval()
        with torch.no_grad():
            val_acc, val_loss = my_fwd_pass(my_X2.view(-1, 1, LEN_OF_INPUT).to(DEVICE), my_y2.to(DEVICE), train=False)
        if(print_now):
            print("Val loss: ", val_loss, "; Val_acc: ", val_acc)
        net.train()
        return val_acc, val_loss
    
    def my_fwd_pass(b_x, b_y, train=False):
        outputs = net(b_x)
        # a matches-t maskepp kene szamolni
        matches = [abs(torch.argmax(i) - torch.argmax(j)) < 0.1 for i, j in zip(outputs, b_y)]
        accuracy = matches.count(True) / len(matches)
        loss = loss_function(outputs, b_y)
        return accuracy, loss
        
    def one_segment_test(start):
        my_X3, my_y3 = TEST_X[start : start + 1], TEST_Y[start : start + 1]
        to_show_wanted = my_y3.to(DEVICE)
        to_show_guessed = net(my_X3.view(-1, 1, LEN_OF_INPUT).to(DEVICE))
        return to_show_wanted, to_show_guessed
    def one_segment_test_on_train(start):
        my_X3, my_y3 = TRAIN_X[start : start + 1], TRAIN_Y[start : start + 1]
        to_show_wanted = my_y3.to(DEVICE)
        to_show_guessed = net(my_X3.view(-1, 1, LEN_OF_INPUT).to(DEVICE))
        return to_show_wanted, to_show_guessed
    
    def create_matrices():
        matrix_test = np.zeros([len(TEST_X), LEN_OF_OUTPUT * 2], dtype=float)  # 2: wan, gue
        for seg in test_segments:
            wanted, guessed = one_segment_test(seg)
            a = wanted.cpu()
            b = guessed.cpu()
            c = a.detach().numpy()
            d = b.detach().numpy()
            matrix_row = np.column_stack([c, d])
            matrix_test[seg] = matrix_row
        matrix_test_l_formed = np.zeros([len(TEST_X), 2, consts.NUM_OF_OUTPUT_DATA_TYPES, consts.LEN_OF_SEGMENTS], dtype=float)
        for test_case in test_segments:
            for j in range(LEN_OF_OUTPUT):
                if j % consts.NUM_OF_OUTPUT_DATA_TYPES == 0:
                    matrix_test_l_formed[test_case][0][0][int(j / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_test[test_case][j]
                if j % consts.NUM_OF_OUTPUT_DATA_TYPES == 1:
                    matrix_test_l_formed[test_case][0][1][int(j / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_test[test_case][j]
            for k in range(LEN_OF_OUTPUT, LEN_OF_OUTPUT * 2):
                if k % consts.NUM_OF_OUTPUT_DATA_TYPES == 0:
                    matrix_test_l_formed[test_case][1][0][int((k-LEN_OF_OUTPUT) / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_test[test_case][k]
                if k % consts.NUM_OF_OUTPUT_DATA_TYPES == 1:
                    matrix_test_l_formed[test_case][1][1][int((k-LEN_OF_OUTPUT) / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_test[test_case][k]
    
        matrix_train = np.zeros([len(TRAIN_X), LEN_OF_OUTPUT * 2], dtype=float)  # 2: wan, gue
        matrix_train_l_formed = np.zeros([len(TRAIN_X), 2, consts.NUM_OF_OUTPUT_DATA_TYPES, consts.LEN_OF_SEGMENTS], dtype=float)
        for seg in train_segments:
            wanted, guessed = one_segment_test_on_train(seg)
            a = wanted.cpu()
            b = guessed.cpu()
            c = a.detach().numpy()
            d = b.detach().numpy()
            matrix_row = np.column_stack([c, d])
            matrix_train[seg] = matrix_row
        for train_case in train_segments:
            for j in range(LEN_OF_OUTPUT):
                if j % consts.NUM_OF_OUTPUT_DATA_TYPES == 0:
                    matrix_train_l_formed[train_case][0][0][int(j / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_train[train_case][j]
                if j % consts.NUM_OF_OUTPUT_DATA_TYPES == 1:
                    matrix_train_l_formed[train_case][0][1][int(j / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_train[train_case][j]
            for k in range(LEN_OF_OUTPUT, LEN_OF_OUTPUT * 2):
                if k % consts.NUM_OF_OUTPUT_DATA_TYPES == 0:
                    matrix_train_l_formed[train_case][1][0][int((k-LEN_OF_OUTPUT) / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_train[train_case][k]
                if k % consts.NUM_OF_OUTPUT_DATA_TYPES == 1:
                    matrix_train_l_formed[train_case][1][1][int((k-LEN_OF_OUTPUT) / consts.NUM_OF_OUTPUT_DATA_TYPES)] = matrix_train[train_case][k]
        
        return matrix_test_l_formed, matrix_train_l_formed
    matrix_test_glob, matrix_train_glob = create_matrices()
    
for test_seg in test_segments:
    if consts.NUM_OF_TESTS < test_seg:
        print("Túl magas a test_seg érték!", test_seg)
    
    to_show_wanted = matrix_test_glob[test_seg, 0]
    to_show_guessed = matrix_test_glob[test_seg, 1]
    tensor_w = torch.from_numpy(to_show_wanted)
    tensor_g = torch.from_numpy(to_show_guessed)
    loss = loss_function(tensor_w, tensor_g)
    print("a", test_seg, "számú test szegmens loss-a:", loss)
    
    plt.figure(100000+figure_const*10000+test_seg*10)
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    
    ax1.plot(to_show_wanted [0, :] * 10, label="Vel wan")     # *10 : skálázás, tanításkor 10-zel osztjuk
    ax1.plot(to_show_guessed[0, :] * 10, label="Vel gue")
    ax1.legend(loc=1)
    
    ax2.plot(to_show_wanted [1, :] / 6, label="AngVel wan")   # /6 : skálázás, tanításkor 6-tal szorozzuk
    ax2.plot(to_show_guessed[1, :] / 6, label="AngVel gue")
    ax2.legend(loc=1)
    
    plt.show()
    
    plt.figure(100000+figure_const*10000+test_seg*10+1)
    plt.title("Velocity")
    plt.xlabel('moment')
    plt.ylabel('raw output')
    plt.plot(to_show_wanted [0, :], label="wan")
    plt.plot(to_show_guessed[0, :], label="gue")
    plt.legend(loc=1)
    plt.axis([0, consts.LEN_OF_SEGMENTS, -3.5, 3.5])
    plt.show()
    
    plt.figure(100000+figure_const*10000+test_seg*10+2)
    plt.title("Angular Velocity")
    plt.xlabel('moment')
    plt.ylabel('raw output')
    plt.plot(to_show_wanted [1, :], label="wan")
    plt.plot(to_show_guessed[1, :], label="gue")
    plt.legend(loc=1)
    plt.axis([0, consts.LEN_OF_SEGMENTS, -3.5, 3.5])
    plt.show()
    
    plt.show()
    
for train_seg in train_segments:
    if consts.NUM_OF_TESTS < train_seg:
        print("Túl magas a train_seg érték!", train_seg)
    
    to_show_wanted = matrix_train_glob[train_seg, 0]
    to_show_guessed = matrix_train_glob[train_seg, 1]
    tensor_w = torch.from_numpy(to_show_wanted)
    tensor_g = torch.from_numpy(to_show_guessed)
    loss = loss_function(tensor_w, tensor_g)
    print("a", train_seg, "számú train szegmens loss-a:", loss)
    
    plt.figure(200000+figure_const*10000+train_seg*100)
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    
    ax1.plot(to_show_wanted [0, :] * 10, label="Vel wan")     # *10 : skálázás, tanításkor 10-zel osztjuk
    ax1.plot(to_show_guessed[0, :] * 10, label="Vel gue")
    ax1.legend(loc=1)
    
    ax2.plot(to_show_wanted [1, :] / 6, label="AngVel wan")   # /6 : skálázás, tanításkor 6-tal szorozzuk
    ax2.plot(to_show_guessed[1, :] / 6, label="AngVel gue")
    ax2.legend(loc=1)
    
    plt.show()
    
    plt.figure(200000+figure_const*10000+train_seg*100+1)
    plt.title("Velocity")
    plt.xlabel('moment')
    plt.ylabel('raw output')
    plt.plot(to_show_wanted [0, :], label="wan")
    plt.plot(to_show_guessed[0, :], label="gue")
    plt.legend(loc=1)
    plt.axis([0, consts.LEN_OF_SEGMENTS, -3.5, 3.5])
    plt.show()
    
    plt.figure(200000+figure_const*10000+train_seg*100+2)
    plt.title("Angular Velocity")
    plt.xlabel('moment')
    plt.ylabel('raw output')
    plt.plot(to_show_wanted [1, :], label="wan")
    plt.plot(to_show_guessed[1, :], label="gue")
    plt.legend(loc=1)
    plt.axis([0, consts.LEN_OF_SEGMENTS, -3.5, 3.5])
    plt.show()
    
    plt.show()
