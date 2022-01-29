"""
@author: Bagoly Zoltán
         zoltan.bagoly@gmail.com
"""
import math
import os
import random
import time
from builtins import print

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import tqdm as my_tqdm

READ_DATA = True
NET_TYPE = "convolutional"  # "convolutional" vagy "fully connected"
MODEL_NAME = f"efop__{int(time.time())}"
LABEL_NAME = ""  # f"cnn_conv2_k=4_k=5_dil=4_lr=0.0005__{int(time.time())}"

log_name = MODEL_NAME
print(MODEL_NAME)
device = torch.device("cpu")
print(device)


# GPU
def on_gpu():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


device = on_gpu()
print(device)

# adat elokeszites
my_training_data = []
my_testing_data = []
LEN_OF_SEGMENTS = 400
LEN_OF_INPUT = 4 * LEN_OF_SEGMENTS
LEN_OF_OUTPUT = 5 * LEN_OF_SEGMENTS
DATA_STRIDE = 10
K_FOLD = 5
k_fold_results = []
i_fold_results = []
TEST_DATA_PART_RECIP = 10


# mat = scipy.io.loadmat("meas01.mat")


class DataRead:

    @staticmethod
    def read_from_raw_data(source, start_of_last):
        print("Data preparation")
        mat = scipy.io.loadmat(source)

        # Kimenetek
        fy_fl = mat["WheForceX_FL"]
        fy_fr = mat["WheForceX_FR"]
        fy_rl = mat["WheForceX_RL"]
        fy_rr = mat["WheForceX_RR"]
        wheel_angle = mat["WheAng"]

        # Bemenetek
        orientation = mat["Ori_Z"]
        pos_x = mat["Pos_X"]
        pos_y = mat["Pos_Y"]
        velocity = mat["Vel"]

        segment = 0
        while segment <= start_of_last:
            # for segment in range(0, start_of_last, DATA_STRIDE):
            # print("segment", segment)
            test_segment = False
            if segment + (LEN_OF_SEGMENTS - DATA_STRIDE) + LEN_OF_SEGMENTS <= start_of_last:
                if segment % (LEN_OF_SEGMENTS * TEST_DATA_PART_RECIP) == 0:
                    test_segment = True
                    #print("test0", segment, (LEN_OF_SEGMENTS * TEST_DATA_PART_RECIP))
                    target = my_testing_data
                    segment += LEN_OF_SEGMENTS - DATA_STRIDE
                else:
                    #print("else", segment, (LEN_OF_SEGMENTS * TEST_DATA_PART_RECIP))
                    target = my_training_data
            else:
                target = my_training_data

            #if test_segment == False:
                #print("segment", segment)
            my_input = []
            my_output = []

            angle = -1 * orientation[segment]

            rand_x = random.randint(-100, 100)
            rand_y = random.randint(-100, 100)
            rand_x = 0
            rand_y = 0

            for i_seg in range(LEN_OF_SEGMENTS):
                # print("i", i)
                # print("segment + i", segment + i)

                # X es Y koordinatak szegmensenkent 0-ba tolasa, iranyba forgatasa
                # print("angle:", angle)
                x_norm = pos_x[segment + i_seg] - (pos_x[segment])
                y_norm = pos_y[segment + i_seg] - (pos_y[segment])
                # print("X norm:", x_norm, "Y norm:", y_norm)
                x_norm_rot = math.cos(angle) * x_norm - math.sin(angle) * y_norm
                y_norm_rot = math.sin(angle) * x_norm + math.cos(angle) * y_norm
                # print("X norm rot:", x_norm_rot, "Y norm rot:", y_norm_rot)

                # normalization between -3 and 3
                my_input.append((x_norm_rot / 100) + rand_x)
                my_input.append((y_norm_rot / 100) + rand_y)
                my_input.append(velocity[segment] / 10)
                my_input.append(orientation[segment])

                my_output.append(fy_fl[segment + i_seg] / 1700)
                my_output.append(fy_fr[segment + i_seg] / 1700)
                my_output.append(fy_rl[segment + i_seg] / 1700)
                my_output.append(fy_rr[segment + i_seg] / 1700)
                my_output.append(wheel_angle[segment + i_seg] * 120)

            if test_segment:
                #print("test1", segment)
                segment += LEN_OF_SEGMENTS - DATA_STRIDE

            # else:
            #     print("train")
            target.append([np.array(my_input), np.array(my_output)])
            # print(segment, "my_out", np.array(my_output), np.array(my_output).shape)
            # print("my_in", np.array(my_output).shape, np.array(my_input))

            segment += DATA_STRIDE

        print("my_testing_data", np.array(my_testing_data).shape)
        print("my_training_data", np.array(my_training_data).shape)
        np.random.shuffle(my_testing_data)
        np.save(f"my_testing_data.npy", my_testing_data)
        np.random.shuffle(my_training_data)
        np.save(f"my_training_data.npy", my_training_data)


dr = DataRead()


# Net convolutional
class NetConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=4, stride=4)
        # nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.conv2 = nn.Conv1d(32, 512, kernel_size=5, dilation=1, padding=2)
        # nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.conv3 = nn.Conv1d(512, 1024, kernel_size=5, dilation=1, padding=2)
        # nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        self.conv4 = nn.Conv1d(1024, 2048, kernel_size=5, dilation=1, padding=2)
        # nn.init.kaiming_uniform_(self.conv4.weight, mode='fan_in', nonlinearity='relu')
        self.conv5 = nn.Conv1d(2048, 2048, kernel_size=5, dilation=1, padding=2)
        # nn.init.kaiming_uniform_(self.conv5.weight, mode='fan_in', nonlinearity='relu')
        self.conv6 = nn.Conv1d(2048, 1024, kernel_size=5, dilation=1, padding=2)
        # nn.init.kaiming_uniform_(self.conv4.weight, mode='fan_in', nonlinearity='relu')
        self.conv7 = nn.Conv1d(1024, 512, kernel_size=5, dilation=1, padding=2)
        # nn.init.kaiming_uniform_(self.conv5.weight, mode='fan_in', nonlinearity='relu')
        self.conv8 = nn.Conv1d(512, 512, kernel_size=5, dilation=1, padding=2)
        # nn.init.kaiming_uniform_(self.conv4.weight, mode='fan_in', nonlinearity='relu')
        self.conv9 = nn.Conv1d(512, 512, kernel_size=5, dilation=1, padding=2)
        # nn.init.kaiming_uniform_(self.conv5.weight, mode='fan_in', nonlinearity='relu')

        x = torch.randn(LEN_OF_INPUT).view(-1, 1, LEN_OF_INPUT)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        # nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        self.fc2 = nn.Linear(512, LEN_OF_OUTPUT)
        # nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def convs(self, x):
        # print("\n")
        # print("0", x[0].shape)
        x = functional.max_pool1d(functional.relu(self.conv1(x)), (3))
        # print("1", x[0].shape)
        x = functional.max_pool1d(functional.relu(self.conv2(x)), (3))
        # print("2", x[0].shape)
        x = functional.max_pool1d(functional.relu(self.conv3(x)), (3))
        # print("3", x[0].shape)
        x = functional.max_pool1d(functional.relu(self.conv4(x)), (3))
        # print("4", x[0].shape)
        x = functional.max_pool1d(functional.relu(self.conv5(x)), (1))
        # print("5", x[0].shape, x[0], x)
        x = functional.max_pool1d(functional.relu(self.conv6(x)), (1))
        # print("6", x[0].shape, x[0], x)
        x = functional.max_pool1d(functional.relu(self.conv7(x)), (3))
        # print("7", x[0].shape, x[0], x)
        x = functional.max_pool1d(functional.relu(self.conv8(x)), (1))
        # print("8", x[0].shape, x[0], x)
        x = functional.max_pool1d(functional.relu(self.conv9(x)), (1))
        # print("9", x[0].shape, x[0], x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1]

        return x

    def forward(self, x):
        # print(self.only_linear)
        x = self.convs(x)
        # print("3", x.shape, x[0], x)

        x = x.view(-1, self._to_linear)
        # print("4", x.shape, x[0], x)

        x = functional.relu(self.fc1(x))
        # print("5", x.shape, x[0], x)

        x = self.fc2(x)
        # print("6", x.shape, x[0], x)

        return x


# Net fully connected
class NetFC(nn.Module):
    def __init__(self):
        super().__init__()

        x = torch.randn(LEN_OF_INPUT).view(-1, 1, LEN_OF_INPUT)
        self._to_linear = None
        self.linearize(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        # nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        self.fc2 = nn.Linear(512, 1024)
        # nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        self.fc3 = nn.Linear(1024, 1024)
        # nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        self.fc4 = nn.Linear(1024, 1024)
        # nn.init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(512, 512)
        self.fc_last = nn.Linear(512, LEN_OF_OUTPUT)
        # nn.init.kaiming_uniform_(self.fc_last.weight, mode='fan_in', nonlinearity='relu')

    def linearize(self, x):
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1]

        return x

    def forward(self, x):
        # print("33", x.shape, x[0], x)

        x = x.view(-1, self._to_linear)
        # print("4", x.shape, x[0], x)

        x = functional.relu(self.fc1(x))
        # print("5", x.shape, x[0], x)

        x = functional.relu(self.fc2(x))
        # print("6", x.shape, x[0], x)

        x = functional.relu(self.fc3(x))
        # print("7", x.shape, x[0], x)

        x = functional.relu(self.fc4(x))
        # print("8", x.shape, x[0], x)

        x = functional.relu(self.fc5(x))
        # print("9", x.shape, x[0], x)

        x = functional.relu(self.fc6(x))
        # print("10", x.shape, x[0], x)

        x = self.fc_last(x)
        # print("last", x.shape, x[0], x)

        return x

if NET_TYPE == "convolutional":
    net = NetConv()
    optimizer = optim.Adam(net.parameters(), lr=0.0005  # , weight_decay=0.1)
                           )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=110, gamma=0.5)
if NET_TYPE == "fully connected":
    net = NetFC()
    optimizer = optim.Adam(net.parameters(), lr=0.0002  # , weight_decay=0.1)
                           )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=110, gamma=0.5)

net.to(device)
loss_function = nn.MSELoss()

my_training_data.clear()
my_testing_data.clear()
i_fold_results.clear()

if READ_DATA:
    # dr.read_from_raw_data("meas01.mat", 10118 - LEN_OF_SEGMENTS)
    # dr.read_from_raw_data("meas02.mat", 10305 - LEN_OF_SEGMENTS)
    dr.read_from_raw_data("data_zoli_carmaker_leaf_02.mat", 137237 - LEN_OF_SEGMENTS)

shuffled_training_data_k = np.load(f"my_training_data_k.npy", allow_pickle=True)
shuffled_testing_data_k = np.load(f"my_testing_data_k.npy", allow_pickle=True)
# print("shu_td", shuffled_training_data_k.shape)


def data_division_and_shaping():
    # print([item[0] for item in shuffled_training_data])
    # print(shuffled_training_data[1][1].shape)
    my_x = torch.Tensor([i_in_shuffled_1[0] for i_in_shuffled_1 in shuffled_training_data_k]).view(-1, LEN_OF_INPUT)
    my_X_t = torch.Tensor([i_in_shuffled_2[0] for i_in_shuffled_2 in shuffled_testing_data_k]).view(-1, LEN_OF_INPUT)
    print("X", my_x.shape)
    my_y = torch.Tensor([i_in_shuffled_3[1] for i_in_shuffled_3 in shuffled_training_data_k]).view(-1, LEN_OF_OUTPUT)
    my_y_t = torch.Tensor([i_in_shuffled_3[1] for i_in_shuffled_3 in shuffled_testing_data_k]).view(-1, LEN_OF_OUTPUT)
    print("y", my_y.shape)

    # needed if the data is not devided already
    # val_size = int(len(my_x) / K_FOLD)
    # print(val_size)

    # print((my_x[0 : (K_FOLD - ite) * val_size]).shape)
    # my_train_X_l = torch.cat((my_x[0 : (K_FOLD - ite) * val_size],
    #                           my_x[((K_FOLD - ite) * val_size) + val_size : K_FOLD * val_size]), 0)
    # my_test_X_l = my_x[(K_FOLD - ite) * val_size : ((K_FOLD - ite) * val_size) + val_size]
    # my_train_y_l = torch.cat((my_y[0 : (K_FOLD - ite) * val_size],
    #                           my_y[((K_FOLD - ite) * val_size) + val_size : K_FOLD * val_size]), 0)
    # my_test_y_l = my_y[(K_FOLD - ite) * val_size : ((K_FOLD - ite) * val_size) + val_size]

    # needed if data is divided in data preparation (to shuffled_training_data and shuffled_testing_data)
    my_test_X_l = my_X_t
    my_test_y_l = my_y_t
    my_train_X_l = my_x
    my_train_y_l = my_y

    print("train_X", my_train_X_l.shape)
    print("train_y", my_train_y_l.shape)
    print("my_test_X", my_test_X_l.shape)
    print("my_test_y", my_test_y_l.shape)

    return my_test_X_l, my_test_y_l, my_train_X_l, my_train_y_l


my_test_X, my_test_y, my_train_X, my_train_y = data_division_and_shaping()


def my_test(size=3, print_now=False):
    # print(my_test_X.shape, my_test_X)
    # print("len(my_test_X)", len(my_test_X))
    # print(size)
    # random_start = np.random.randint(len(my_test_X) - size)
    start = 0

    # my_X2, my_y2 = my_test_X[random_start: random_start + size], my_test_y[random_start: random_start + size]
    my_X2, my_y2 = my_test_X[ : ], my_test_y[ : ]
    net.eval()
    with torch.no_grad():
        val_acc_test, val_loss_test = my_fwd_pass(my_X2.view(-1, 1, LEN_OF_INPUT).to(device), my_y2.to(device), train=False)
    if print_now:
        print("Val loss: ", val_loss_test, "; Val_acc: ", val_acc_test)
    net.train()
    return val_acc_test, val_loss_test


def my_fwd_pass(b_x, b_y, train=False):
    if train:
        net.zero_grad()
    # print("b_x", b_x.shape)
    # print("b_y:", b_y.shape)
    outputs = net(b_x)
    # print("outputs: ", outputs.shape, outputs)
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
    #     file.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(outputs),10)}\n")
    #     file.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(loss),15)}\n")
    return accuracy, loss


def my_train():
    BATCH_SIZE = 16
    EPOCHS = 300
    cnt = 2
    print("Training starts")
    with open("efop_log.log", "a") as file_train:
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch)
            for i in my_tqdm.tqdm(range(0, len(my_train_X), BATCH_SIZE)):
                my_batch_x = my_train_X[i: i + BATCH_SIZE].view(-1, 1, LEN_OF_INPUT).to(device)
                my_batch_y = my_train_y[i: i + BATCH_SIZE].to(device)
                # print("batch_x", my_batch_x.shape, my_batch_x)
                # print("batch_y", my_batch_y)
                # print("train_x", my_train_X.shape)
                # print("train_y", my_train_y.shape)
                accuracy, loss = my_fwd_pass(my_batch_x, my_batch_y, train=True)
                if i % 50 == 0:
                    val_acc_2, val_loss_2 = my_test()
                    cnt += 1
                    file_train.write(
                        f"{log_name},{cnt},{round(float(accuracy), 5)},{round(float(loss), 10)},{round(float(val_acc_2), 5)},{round(float(val_loss_2), 10)}\n")
            my_test(print_now=True)
            scheduler.step()
    for tests in range(5):
        val_acc_1, val_loss_1 = my_test()
        i_fold_results.append(val_loss_1)
        print("i_fold_results:", i_fold_results)


# Tanitas nelkuli halo valasza
with open("efop_log.log", "a") as file:
    # file.write(f"{log_name},0,{round(float(3200),5)},3200,{round(float(3200),5)},3200\n")
    val_acc, val_loss = my_test()
    print("Tanitas nelkuli halo valaszan a kulonbseg:", val_loss)
    file.write(
        f"{log_name},1,{round(float(val_acc), 5)},{round(float(val_loss), 10)},{round(float(val_acc), 5)},{round(float(val_loss), 10)}\n")

my_train()
# print(sum(i_fold_results))
# print(len(i_fold_results))
# print(sum(i_fold_results) / len(i_fold_results))
k_fold_results.append(sum(i_fold_results) / len(i_fold_results))

path = os.path.join('net_{}.pth'.format(MODEL_NAME))
torch.save(net, path)

# Vizualizalasa a tanitott halo mukodesenek
model_name = MODEL_NAME


def create_acc_loss_graph(model_name_p):
    contents = open("efop_log.log", "r").read().split('\n')
    times = []
    accuracies = []
    losses = []
    val_accuracies = []
    val_losses = []
    too_highs = 0

    for c in contents:
        if model_name_p in c:
            name, timestamp, acc, loss, val_acc_graph, val_loss_graph = c.split(",")

            if float(loss) < 3000:
                if float(val_loss_graph) < 3000:
                    times.append(float(timestamp))
                    accuracies.append(float(acc))
                    losses.append(float(loss))
                    val_accuracies.append(float(val_acc_graph))
                    val_losses.append(float(val_loss_graph))
                else:
                    too_highs += 1
            else:
                too_highs += 1

    print("too_highs: ", too_highs)
    fig = plt.figure()
    # ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0)  # , sharex=ax1)
                           )
    # ax1.plot(times, accuracies, label="Accuracies")
    # ax1.plot(times, val_accuracies, label="Val_Accuracies")
    # ax1.legend(loc=2)

    ax2.plot(times, losses, 'orange', label=LABEL_NAME)
    ax2.plot(times, val_losses, 'blue')
    ax2.legend(loc=1)

    plt.show()


create_acc_loss_graph(model_name)


def one_segment_test(start):
    # print(start)
    my_X3, my_y3 = my_test_X[start: start + 1], my_test_y[start: start + 1]
    to_show_wanted = my_y3.to(device)
    to_show_guessed = net(my_X3.view(-1, 1, LEN_OF_INPUT).to(device))
    # print(to_show_wanted.shape)
    # print(to_show_guessed.shape)
    # print(" Target:", to_show_wanted, "\n", "Guess:", to_show_guessed)
    return to_show_wanted, to_show_guessed


def lets_see(to_show_wanted_, to_show_guessed_):
    FL_w = []
    FR_w = []
    RL_w = []
    RR_w = []
    SW_w = []

    FL_g = []
    FR_g = []
    RL_g = []
    RR_g = []
    SW_g = []

    FL_w_ = []
    FR_w_ = []
    RL_w_ = []
    RR_w_ = []
    SW_w_ = []

    FL_g_ = []
    FR_g_ = []
    RL_g_ = []
    RR_g_ = []
    SW_g_ = []

    for i in range(LEN_OF_OUTPUT):
        if (i % 5) == 0:
            FL_w.append(float(to_show_wanted_[0][i]) * 1700)
            FL_g.append(float(to_show_guessed_[0][i]) * 1700)
            FL_w_.append(float(to_show_wanted_[0][i]))
            FL_g_.append(float(to_show_guessed_[0][i]))
        if (i % 5) == 1:
            FR_w.append(float(to_show_wanted_[0][i]) * 1700)
            FR_g.append(float(to_show_guessed_[0][i]) * 1700)
            FR_w_.append(float(to_show_wanted_[0][i]))
            FR_g_.append(float(to_show_guessed_[0][i]))
        if (i % 5) == 2:
            RL_w.append(float(to_show_wanted_[0][i]) * 1700)
            RL_g.append(float(to_show_guessed_[0][i]) * 1700)
            RL_w_.append(float(to_show_wanted_[0][i]))
            RL_g_.append(float(to_show_guessed_[0][i]))
        if (i % 5) == 3:
            RR_w.append(float(to_show_wanted_[0][i]) * 1700)
            RR_g.append(float(to_show_guessed_[0][i]) * 1700)
            RR_w_.append(float(to_show_wanted_[0][i]))
            RR_g_.append(float(to_show_guessed_[0][i]))
        if (i % 5) == 4:
            SW_w.append(float(to_show_wanted_[0][i]) / 120)
            SW_g.append(float(to_show_guessed_[0][i]) / 120)
            SW_w_.append(float(to_show_wanted_[0][i]))
            SW_g_.append(float(to_show_guessed_[0][i]))

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


with open("diff_avgs.log", "a") as file:
    file.write(f"net: {path} \n")
    for test in range(0, len(my_test_X), 1):
        sum_diff_SW = 0
        avg_diff_SW = 0
        sum_diff_FA = 0
        avg_diff_FA = 0
        sum_diff_RA = 0
        avg_diff_RA = 0
        wanted, guessed = one_segment_test(test)
        for moment in range(LEN_OF_OUTPUT):
            if (moment % 5) == 4:
                sum_diff_SW += abs((float(guessed[0][moment]) / 120) - (float(wanted[0][moment]) / 120))
            if (moment % 5) == 0:
                sum_diff_FA += abs((float(guessed[0][moment]) * 1700) - (float(wanted[0][moment]) * 1700))
            if (moment % 5) == 1:
                sum_diff_FA += abs((float(guessed[0][moment]) * 1700) - (float(wanted[0][moment]) * 1700))
            if (moment % 5) == 2:
                sum_diff_RA += abs((float(guessed[0][moment]) * 1700) - (float(wanted[0][moment]) * 1700))
            if (moment % 5) == 3:
                sum_diff_RA += abs((float(guessed[0][moment]) * 1700) - (float(wanted[0][moment]) * 1700))
        avg_diff_SW = sum_diff_SW / LEN_OF_SEGMENTS
        avg_diff_FA = sum_diff_FA / LEN_OF_SEGMENTS
        avg_diff_RA = sum_diff_RA / LEN_OF_SEGMENTS
        file.write(f"{test}\t Ave. diff SW: {round(float(avg_diff_SW), 4)}\t Ave. diff FA: {round(float(avg_diff_FA), 1)}\t Ave. diff RA: {round(float(avg_diff_RA), 1)} \n")

for test in range(0, len(my_test_X), 1):
    to_show_wanted_glob, to_show_guessed_glob = one_segment_test(test)  # (np.random.randint(len(my_test_X) - 1))
    lets_see(to_show_wanted_glob, to_show_guessed_glob)

# to_show_wanted_glob_array = to_show_wanted_glob.cpu().detach().numpy()
#to_show_guessed_glob_array = to_show_guessed_glob.cpu().detach().numpy()
#counted_MSE, counted_error = 0, 0
#for i in range(0, len(to_show_guessed_glob_array[0])):
#    counted_MSE += math.pow((to_show_wanted_glob_array[0][i] - to_show_guessed_glob_array[0][i]), 2)
#    counted_error += abs(to_show_wanted_glob_array[0][i] - to_show_guessed_glob_array[0][i])
#counted_MSE = counted_MSE / len(to_show_guessed_glob_array[0])
#counted_error = counted_error / len(to_show_guessed_glob_array[0])
# print("")
# print("Counted MSE = ", counted_MSE)
# print("Counted mean error =", counted_error)
# given_MSE = loss_function(to_show_guessed_glob, to_show_wanted_glob)
# print("Given MSE = ", given_MSE)


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
        # print(seg)
        wanted_plot, guessed_plot = one_segment_test(seg)
        # diff_pct = 0
        diff_val_SW = 0
        diff_val_SW_2 = 0
        diff_val_front = 0
        diff_val_rear = 0
        diff_val_front_br = 0
        diff_val_front_acc = 0
        diff_val_rear_br = 0
        diff_val_rear_acc = 0
        for moment_plot in range(LEN_OF_OUTPUT):
            if (moment_plot % 5) == 4:
                diff_val_SW += math.pow((float(guessed_plot[0][moment_plot]) / 1) - (float(wanted_plot[0][moment_plot]) / 1), 2)
                diff_val_SW_2 += abs((float(guessed_plot[0][moment_plot]) / 120) - (float(wanted_plot[0][moment_plot]) / 120))
            if (moment_plot % 5) == 0:
                diff_val_front += math.pow((float(guessed_plot[0][moment_plot]) * 1) - (float(wanted_plot[0][moment_plot])) * 1, 2)
                if float(wanted_plot[0][moment_plot]) < 0:
                    diff_val_front_br += abs((float(guessed_plot[0][moment_plot]) * 1700) - (float(wanted_plot[0][moment_plot])) * 1700)
                else:
                    diff_val_front_acc += abs(
                        (float(guessed_plot[0][moment_plot]) * 1700) - (float(wanted_plot[0][moment_plot])) * 1700)
            if (moment_plot % 5) == 1:
                diff_val_front += math.pow((float(guessed_plot[0][moment_plot]) * 1) - (float(wanted_plot[0][moment_plot])) * 1, 2)
                if float(wanted_plot[0][moment_plot]) < 0:
                    diff_val_front_br += abs((float(guessed_plot[0][moment_plot]) * 1700) - (float(wanted_plot[0][moment_plot])) * 1700)
                else:
                    diff_val_front_acc += abs(
                        (float(guessed_plot[0][moment_plot]) * 1700) - (float(wanted_plot[0][moment_plot])) * 1700)
            if (moment_plot % 5) == 2:
                diff_val_rear += math.pow((float(guessed_plot[0][moment_plot]) * 1) - (float(wanted_plot[0][moment_plot])) * 1, 2)
                if float(wanted_plot[0][moment_plot]) < 0:
                    diff_val_rear_br += abs((float(guessed_plot[0][moment_plot]) * 1700) - (float(wanted_plot[0][moment_plot])) * 1700)
                else:
                    diff_val_rear_acc += abs((float(guessed_plot[0][moment_plot]) * 1700) - (float(wanted_plot[0][moment_plot])) * 1700)
            if (moment_plot % 5) == 3:
                diff_val_rear += math.pow((float(guessed_plot[0][moment_plot]) * 1) - (float(wanted_plot[0][moment_plot])) * 1, 2)
                if float(wanted_plot[0][moment_plot]) < 0:
                    diff_val_rear_br += abs((float(guessed_plot[0][moment_plot]) * 1700) - (float(wanted_plot[0][moment_plot])) * 1700)
                else:
                    diff_val_rear_acc += abs((float(guessed_plot[0][moment_plot]) * 1700) - (float(wanted_plot[0][moment_plot])) * 1700)
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
    # print("losses_val_SW [0]: ", losses_val_SW[0])
    # print("losses_val_front [0]: ", losses_val_front[0])
    # print("losses_val_rear [0]: ", losses_val_rear[0])

    # print(losses_pct)

    fig = plt.figure()

    ax1 = plt.subplot2grid((3, 1), (0, 0))
    # ax1.plot(segments, losses_pct, label="% diff per segments in testdata")
    ax1.plot(losses_val_SW, 'b', label="val diff per seg in tested SWA")
    ax1.legend(loc=2)

    ax2 = plt.subplot2grid((3, 1), (1, 0))
    ax2.plot(losses_val_front, 'b', label="val diff per seg in tested WF front wheels")
    ax2.legend(loc=2)

    ax3 = plt.subplot2grid((3, 1), (2, 0))
    ax3.plot(losses_val_rear, 'b', label="val diff per seg in tested WF rear wheels")
    ax3.legend(loc=2)

    plt.show()

    fig = plt.figure()

    ax1 = plt.subplot2grid((5, 1), (0, 0))
    ax1.plot(losses_val_front_br, 'b', label="val diff per seg in tested WF front wheels while braking")
    ax1.legend(loc=2)

    ax2 = plt.subplot2grid((5, 1), (1, 0))
    ax2.plot(losses_val_front_acc, 'b', label="val diff per seg in tested WF front wheels while accelerating")
    ax2.legend(loc=2)

    ax3 = plt.subplot2grid((5, 1), (2, 0))
    ax3.plot(losses_val_rear_br, 'b', label="val diff per seg in tested WF rear wheels while braking")
    ax3.legend(loc=2)

    ax4 = plt.subplot2grid((5, 1), (3, 0))
    ax4.plot(losses_val_rear_acc, 'b', label="val diff per seg in tested WF rear wheels while accelerating")
    ax4.legend(loc=2)

    ax1 = plt.subplot2grid((5, 1), (4, 0))
    ax1.plot(losses_val_SW_2, 'b', label="val diff per seg in tested SW")
    ax1.legend(loc=2)

    plt.show()


plot_losses_after_training()
