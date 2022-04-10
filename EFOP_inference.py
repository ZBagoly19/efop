# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 12:22:04 2021

@author: Bagoly Zoltán
"""

import os
import math
import numpy as np
import tqdm as my_tqdm
import matplotlib.pyplot as plt
from matplotlib import style
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

NET_TYPE =   "fully connected"   # "convolutional" or "fully connected"
TIME__ = 1647769026
epoch = 300
TRAINING_NAME = "efop__" + str(TIME__)
LABEL_NAME = f"fc__{int(time.time())}"
print("")
print(TRAINING_NAME)
device = torch.device("cpu")

# adat elokeszites
LEN_OF_SEGMENTS = 400
LEN_OF_INPUT = 4 * LEN_OF_SEGMENTS
LEN_OF_OUTPUT = 5 * LEN_OF_SEGMENTS
DATA_STRIDE = 10

shuffled_training_data = np.load("my_training_data_1d_XYVO_better_skale.npy", allow_pickle=True)
shuffled_testing_data = np.load("my_testing_data_1d_XYVO_better_skale.npy", allow_pickle=True)
my_X = torch.Tensor([i[0] for i in shuffled_training_data]).view(-1, LEN_OF_INPUT)
my_X_t = torch.Tensor([i[0] for i in shuffled_testing_data]).view(-1, LEN_OF_INPUT)
my_y = torch.Tensor([i[1] for i in shuffled_training_data]).view(-1, LEN_OF_OUTPUT)
my_y_t = torch.Tensor([i[1] for i in shuffled_testing_data]).view(-1, LEN_OF_OUTPUT)
my_test_X_l = my_X_t
my_test_y_l = my_y_t
my_train_X_l = my_X
my_train_y_l = my_y

my_test_X, my_test_y, my_train_X, my_train_y = my_test_X_l, my_test_y_l, my_train_X_l, my_train_y_l

matrix = [None] * len(my_test_X)

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
        
        self.fc1 = nn.Linear(self._to_linear, 2048)
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        self.fc2 = nn.Linear(2048, 2048)
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        self.fc3 = nn.Linear(2048, 2048)
        nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        self.fc4 = nn.Linear(2048, 2048)
        nn.init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
        self.fc5 = nn.Linear(2048, 2048)
        nn.init.kaiming_uniform_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
        self.fc6 = nn.Linear(2048, 2048)
        nn.init.kaiming_uniform_(self.fc6.weight, mode='fan_in', nonlinearity='relu')
        # self.fc7 = nn.Linear(2048, 2048)
        # nn.init.kaiming_uniform_(self.fc7.weight, mode='fan_in', nonlinearity='relu')
        # self.fc8 = nn.Linear(2048, 2048)
        # nn.init.kaiming_uniform_(self.fc8.weight, mode='fan_in', nonlinearity='relu')
        # self.fc9 = nn.Linear(2048, 2048)
        # nn.init.kaiming_uniform_(self.fc9.weight, mode='fan_in', nonlinearity='relu')
        # self.fc10 = nn.Linear(2048, 2048)
        # nn.init.kaiming_uniform_(self.fc10.weight, mode='fan_in', nonlinearity='relu')
        self.fc_last = nn.Linear(2048, LEN_OF_OUTPUT)
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
        #print("9", x.shape, x[0], x)
        
        x = F.relu(self.fc6(x))
        #print("10", x.shape, x[0], x)
        
        # x = F.relu(self.fc7(x))
        # #print("11", x.shape, x[0], x)
        
        # x = F.relu(self.fc8(x))
        # #print("12", x.shape, x[0], x)
        
        # x = F.relu(self.fc9(x))
        # #print("13", x.shape, x[0], x)
        
        # x = F.relu(self.fc10(x))
        # #print("14", x.shape, x[0], x)
        
        x = self.fc_last(x)
        # print("last", x.shape, x[0], x)
        
        return x

def load_net(path_param):
    if os.path.isfile(path_param):
        print("helyes filenev:", path_param)
        
        net = torch.load(os.path.join(path_param))
        print(net)
        net.eval()

loss_function = nn.MSELoss()
MODEL_NUMBER = 1
print('net_{}.pth'.format(TIME__ *100000 + MODEL_NUMBER *1000 + epoch))
while os.path.isfile(os.path.join('net_{}.pth'.format(TIME__ *100000 + MODEL_NUMBER *1000 + epoch))):
    print(os.path.join('net_{}.pth'.format(TIME__ *100000 + MODEL_NUMBER *1000 + epoch)))
    
    net = torch.load(os.path.join('net_{}.pth'.format(TIME__ *100000 + MODEL_NUMBER *1000 + epoch)))
    net.eval()
    
    def my_test(size=64, print_now=False):
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
    
        # with open("efop_log_output_of_net.log", "a") as file:
        #     # for i in range(len(outputs)):
        #     #     output = outputs[0][i]
        #     file.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(outputs),10)}\n")
        #     file.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(loss),15)}\n")
        return accuracy, loss
    
    # Vizualizalasa a tanitott halo mukodesenek
    style.use("ggplot")
    model_name = TRAINING_NAME
    def create_acc_loss_graph(model_name):
        contents = open("efop_log.log", "r").read().split('\n')
        times = []
        accuracies = []
        losses = []
        val_accuracies = []
        val_losses = []
        too_highs = 0
        
        for c in contents:
            if model_name in c:
                name, timestamp, acc, loss, val_acc, val_loss = c.split(",")
                
                if float(loss) < 80:
                    if float(val_loss) < 80:
                        times.append(float(timestamp))
                        accuracies.append(float(acc))
                        losses.append(float(loss))
                        val_accuracies.append(float(val_acc))
                        val_losses.append(float(val_loss))
                    else:
                        too_highs += 1
                else:
                    too_highs += 1

        print("too_highs: ", too_highs)
        fig = plt.figure()
        #ax1 = plt.subplot2grid((2, 1), (0, 0))
        ax2 = plt.subplot2grid((2, 1), (1, 0)#, sharex=ax1)
        )
        # ax1.plot(times, accuracies, label="Accuracies")
        # ax1.plot(times, val_accuracies, label="Val_Accuracies")
        # ax1.legend(loc=2)
        
        ax2.plot(times, losses, label=TIME__ *100000 + MODEL_NUMBER *1000 + epoch)
        ax2.plot(times, val_losses, 'b')
        ax2.legend(loc=1)
        
        plt.show()
        
    create_acc_loss_graph(model_name)
    
    def one_segment_test(start):
        # print(start)
        my_X3, my_y3 = my_test_X[start : start + 1], my_test_y[start : start + 1]
        to_show_wanted = my_y3.to(device)
        to_show_guessed = net(my_X3.view(-1, 1, LEN_OF_INPUT).to(device))
        # print(to_show_wanted.shape)
        # print(to_show_guessed.shape)
        # print(" Target:", to_show_wanted, "\n", "Guess:", to_show_guessed)
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
    
    def lets_see(to_show_wanted_, to_show_guessed_):
        FL_w = [None] * LEN_OF_SEGMENTS
        FR_w = [None] * LEN_OF_SEGMENTS
        RL_w = [None] * LEN_OF_SEGMENTS
        RR_w = [None] * LEN_OF_SEGMENTS
        SW_w = [None] * LEN_OF_SEGMENTS
        
        FL_g = [None] * LEN_OF_SEGMENTS
        FR_g = [None] * LEN_OF_SEGMENTS
        RL_g = [None] * LEN_OF_SEGMENTS
        RR_g = [None] * LEN_OF_SEGMENTS
        SW_g = [None] * LEN_OF_SEGMENTS
        
        FL_w_ = [None] * LEN_OF_SEGMENTS
        FR_w_ = [None] * LEN_OF_SEGMENTS
        RL_w_ = [None] * LEN_OF_SEGMENTS
        RR_w_ = [None] * LEN_OF_SEGMENTS
        SW_w_ = [None] * LEN_OF_SEGMENTS
        
        FL_g_ = [None] * LEN_OF_SEGMENTS
        FR_g_ = [None] * LEN_OF_SEGMENTS
        RL_g_ = [None] * LEN_OF_SEGMENTS
        RR_g_ = [None] * LEN_OF_SEGMENTS
        SW_g_ = [None] * LEN_OF_SEGMENTS
            
        place = 0
        for i in range(LEN_OF_OUTPUT):
            if (i % 5) == 0:
                FL_w[place] = float(to_show_wanted_[0][i]) * 1700
                FL_g[place] = float(to_show_guessed_[0][i]) * 1700
                FL_w_[place] = float(to_show_wanted_[0][i])
                FL_g_[place] = float(to_show_guessed_[0][i])
            if (i % 5) == 1:
                FR_w[place] = float(to_show_wanted_[0][i]) * 1700
                FR_g[place] = float(to_show_guessed_[0][i]) * 1700
                FR_w_[place] = float(to_show_wanted_[0][i])
                FR_g_[place] = float(to_show_guessed_[0][i])
            if (i % 5) == 2:    
                RL_w[place] = float(to_show_wanted_[0][i]) * 1700
                RL_g[place] = float(to_show_guessed_[0][i]) * 1700
                RL_w_[place] = float(to_show_wanted_[0][i])
                RL_g_[place] = float(to_show_guessed_[0][i])
            if (i % 5) == 3:
                RR_w[place] = float(to_show_wanted_[0][i]) * 1700
                RR_g[place] = float(to_show_guessed_[0][i]) * 1700
                RR_w_[place] = float(to_show_wanted_[0][i])
                RR_g_[place] = float(to_show_guessed_[0][i])
            if (i % 5) == 4:
                SW_w[place] = float(to_show_wanted_[0][i]) / 60
                SW_g[place] = float(to_show_guessed_[0][i]) / 60
                SW_w_[place] = float(to_show_wanted_[0][i])
                SW_g_[place] = float(to_show_guessed_[0][i])
                place += 1
                
        fig = plt.figure()
        ax1 = plt.subplot2grid((5, 1), (0, 0))
        ax2 = plt.subplot2grid((5, 1), (1, 0))
        ax3 = plt.subplot2grid((5, 1), (2, 0))
        ax4 = plt.subplot2grid((5, 1), (3, 0))
        ax5 = plt.subplot2grid((5, 1), (4, 0))
                               
        ax1.plot(FL_w, label="FL wanted")
        ax1.plot(FL_g, label="FL guessed")
        ax1.legend(loc=1)
        
        ax2.plot(FR_w, label="FR wanted")
        ax2.plot(FR_g, label="FR guessed")
        ax2.legend(loc=1)
        
        ax3.plot(RL_w, label="RL wanted")
        ax3.plot(RL_g, label="RL guessed")
        ax3.legend(loc=1)
        
        ax4.plot(RR_w, label="RR wanted")
        ax4.plot(RR_g, label="RR guessed")
        ax4.legend(loc=1)
        
        ax5.plot(SW_w, label="SW wanted")
        ax5.plot(SW_g, label="SW guessed")
        ax5.legend(loc=1)
        
        plt.show()
        
        fig = plt.figure()
        ax1 = plt.subplot2grid((5, 1), (0, 0))
        ax2 = plt.subplot2grid((5, 1), (1, 0))
        ax3 = plt.subplot2grid((5, 1), (2, 0))
        ax4 = plt.subplot2grid((5, 1), (3, 0))
        ax5 = plt.subplot2grid((5, 1), (4, 0))
                               
        ax1.plot(FL_w_, label="FL wanted")
        ax1.plot(FL_g_, label="FL guessed")
        ax1.legend(loc=1)
        
        ax2.plot(FR_w_, label="FR wanted")
        ax2.plot(FR_g_, label="FR guessed")
        ax2.legend(loc=1)
        
        ax3.plot(RL_w_, label="RL wanted")
        ax3.plot(RL_g_, label="RL guessed")
        ax3.legend(loc=1)
        
        ax4.plot(RR_w_, label="RR wanted")
        ax4.plot(RR_g_, label="RR guessed")
        ax4.legend(loc=1)
        
        ax5.plot(SW_w_, label="SW wanted")
        ax5.plot(SW_g_, label="SW guessed")
        ax5.legend(loc=1)
        
        plt.show()
        
    # for i in range(30):
    #     to_show_wanted_glob, to_show_guessed_glob = one_segment_test(i)#(np.random.randint(len(my_test_X) - 1))
    #     lets_see(to_show_wanted_glob, to_show_guessed_glob)
    #
    # to_show_wanted_glob_array = to_show_wanted_glob.cpu().detach().numpy()
    # to_show_guessed_glob_array = to_show_guessed_glob.cpu().detach().numpy()
    # counted_MSE, counted_error = 0, 0
    # for i in range(0, len(to_show_guessed_glob_array[0])):
    #     counted_MSE += math.pow((to_show_wanted_glob_array[0][i] - to_show_guessed_glob_array[0][i]), 2)
    #     counted_error += abs(to_show_wanted_glob_array[0][i] - to_show_guessed_glob_array[0][i])
    # counted_MSE = counted_MSE / len(to_show_guessed_glob_array[0])
    # counted_error = counted_error / len(to_show_guessed_glob_array[0])
    # print("")
    # #print("Counted MSE = ", counted_MSE)
    # #print("Counted mean error =", counted_error)
    # given_MSE = loss_function(to_show_guessed_glob, to_show_wanted_glob)
    # print("MSE = ", given_MSE)
    
    def plot_losses_after_training():
        # losses_pct = []
        losses_val_SW = []
        losses_val_SW_2 = []
        losses_val_front = []
        losses_val_rear = []
        losses_val_front_br = []
        losses_val_rear_br = []
        losses_val_front_acc = []
        losses_val_rear_acc = []
        segments = []
        # print(len(my_test_X))
        # print(range(len(my_test_X)))
        for seg in my_tqdm.tqdm(range(0, len(my_test_X), 1)):
            #print(seg)
            wanted, guessed = one_segment_test(seg)
            matrix[seg] = [wanted, guessed]
            lets_see(matrix[seg][0], matrix[seg][1])
            # diff_pct = 0
            diff_val_SW = 0
            diff_val_SW_2 = 0
            diff_val_front = 0
            diff_val_rear = 0
            diff_val_front_br = 0
            diff_val_front_acc = 0
            diff_val_rear_br = 0
            diff_val_rear_acc = 0
            for moment in range(LEN_OF_OUTPUT):
                if (moment % 5) == 4:
                    diff_val_SW += math.pow((float(guessed [0][moment]) / 1) - (float(wanted [0][moment]) / 1), 2)
                    diff_val_SW_2 += math.pow((float(guessed [0][moment]) / 1) - (float(wanted [0][moment]) / 1), 2)
                if (moment % 5) == 0:
                    diff_val_front += math.pow((float(guessed [0][moment]) * 1) - (float(wanted [0][moment])) * 1, 2)
                    if float(wanted [0][moment]) < 0:
                        diff_val_front_br += math.pow((float(guessed [0][moment]) * 1) - (float(wanted [0][moment])) * 1, 2)
                    else:
                        diff_val_front_acc += math.pow((float(guessed [0][moment]) * 1) - (float(wanted [0][moment])) * 1, 2)
                if (moment % 5) == 1:
                    diff_val_front += math.pow((float(guessed [0][moment]) * 1) - (float(wanted [0][moment])) * 1, 2)
                    if float(wanted [0][moment]) < 0:
                        diff_val_front_br += math.pow((float(guessed [0][moment]) * 1) - (float(wanted [0][moment])) * 1, 2)
                    else:
                        diff_val_front_acc += math.pow((float(guessed [0][moment]) * 1) - (float(wanted [0][moment])) * 1, 2)
                if (moment % 5) == 2:
                    diff_val_rear += math.pow((float(guessed [0][moment]) * 1) - (float(wanted [0][moment])) * 1, 2)
                    if float(wanted [0][moment]) < 0:
                        diff_val_rear_br += math.pow((float(guessed [0][moment]) * 1) - (float(wanted [0][moment])) * 1, 2)
                    else:
                        diff_val_rear_acc += math.pow((float(guessed [0][moment]) * 1) - (float(wanted [0][moment])) * 1, 2)
                if (moment % 5) == 3:
                    diff_val_rear += math.pow((float(guessed [0][moment]) * 1) - (float(wanted [0][moment])) * 1, 2)
                    if float(wanted [0][moment]) < 0:
                        diff_val_rear_br += math.pow((float(guessed [0][moment]) * 1) - (float(wanted [0][moment])) * 1, 2)
                    else:
                        diff_val_rear_acc += math.pow((float(guessed [0][moment]) * 1) - (float(wanted [0][moment])) * 1, 2)
                # diff_pct += abs(100  -  ((float(guessed [0][moment])) / (float(wanted [0][moment])) * 100))
            # losses_pct.append(diff_pct / LEN_OF_OUTPUT)
            losses_val_SW.append(diff_val_SW / LEN_OF_SEGMENTS)
            losses_val_SW_2.append(diff_val_SW_2 / LEN_OF_SEGMENTS)
            losses_val_front.append(diff_val_front / LEN_OF_SEGMENTS / 2)
            losses_val_rear.append(diff_val_rear / LEN_OF_SEGMENTS / 2)
            losses_val_front_br.append(diff_val_front_br / LEN_OF_SEGMENTS)
            losses_val_rear_br.append(diff_val_rear_br / LEN_OF_SEGMENTS)
            losses_val_front_acc.append(diff_val_front_acc / LEN_OF_SEGMENTS)
            losses_val_rear_acc.append(diff_val_rear_acc / LEN_OF_SEGMENTS)
            segments.append(seg)
        #print("losses_val_SW [0]: ", losses_val_SW[0])
        #print("losses_val_front [0]: ", losses_val_front[0])
        #print("losses_val_rear [0]: ", losses_val_rear[0])
        
        # print(losses_pct)
    
        fig = plt.figure()
    
        ax1 = plt.subplot2grid((3, 1), (0, 0))
        # ax1.plot(segments, losses_pct, label="% diff per segments in testdata")
        ax1.plot(losses_val_SW, 'b', label="MSE per seg in testd SWA")
        ax1.legend(loc=2)
        
        ax2 = plt.subplot2grid((3, 1), (1, 0))
        ax2.plot(losses_val_front, 'b', label="MSE per seg in testd WF front wheels")
        ax2.legend(loc=2)
        
        ax3 = plt.subplot2grid((3, 1), (2, 0))
        ax3.plot(losses_val_rear, 'b', label="MSE per seg in testd WF rear wheels")
        ax3.legend(loc=2)
        
        plt.show()
        
        fig = plt.figure()
    
        ax1 = plt.subplot2grid((5, 1), (0, 0))
        ax1.plot(losses_val_front_br, 'b', label="MSE per seg in testd WF front wheels while braking")
        ax1.legend(loc=2)
        
        ax2 = plt.subplot2grid((5, 1), (1, 0))
        ax2.plot(losses_val_front_acc, 'b', label="MSE per seg in testd WF front wheels while accelerating")
        ax2.legend(loc=2)
        
        ax3 = plt.subplot2grid((5, 1), (2, 0))
        ax3.plot(losses_val_rear_br, 'b', label="MSE per seg in testd WF rear wheels while braking")
        ax3.legend(loc=2)
        
        ax4 = plt.subplot2grid((5, 1), (3, 0))
        ax4.plot(losses_val_rear_acc, 'b', label="MSE per seg in testd WF rear wheels while accelerating")
        ax4.legend(loc=2)
        
        ax5 = plt.subplot2grid((5, 1), (4, 0))
        ax5.plot(losses_val_SW_2, 'b', label="MSE per seg in testd SW")
        ax5.legend(loc=2)
        
        plt.show()
        
        
    plot_losses_after_training()
    epoch += 1000
