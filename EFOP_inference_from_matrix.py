# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:38:30 2022

@author: Bagoly Zoltán
"""
import numpy as gfg
import math
import tqdm as my_tqdm
import matplotlib.pyplot as plt
from matplotlib import style
import torch
import torch.nn as nn
import time

NET_TYPE =   "fully connected"   # "convolutional" or "fully connected"
TIME__ = 1649549122   # 1649466572 1649549122
epoch = 399
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
MODEL_NUMBER = 1

losses = []
val_losses = []
loss_function = nn.MSELoss()

# retrieving data from file.
loaded_matrix_test = gfg.loadtxt("matrix_test.txt")
loaded_matrix_train = gfg.loadtxt("matrix_train.txt")


style.use("ggplot")
model_name = TRAINING_NAME
def create_acc_loss_graph(model_name):
    contents = open("efop_log.log", "r").read().split('\n')
    times = []
    accuracies = []
    val_accuracies = []
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
    plt.axis([0, 400, 0, 3])
    plt.show()
    
create_acc_loss_graph(model_name)

def one_segment_test(start, print_loss=False):
    # print(start)
    to_show_wanted = loaded_matrix_test[start][0 : LEN_OF_OUTPUT]
    to_show_guessed = loaded_matrix_test[start][LEN_OF_OUTPUT : ]
    # print(to_show_wanted.shape)
    # print(to_show_guessed.shape)
    # print(" Target:", to_show_wanted, "\n", "Guess:", to_show_guessed)
    if print_loss:
        loss = loss_function(to_show_guessed, to_show_wanted)
        print(loss)
    return to_show_wanted, to_show_guessed
def one_segment_test_on_train(start, print_loss=False):
    # print(start)
    to_show_wanted = loaded_matrix_train[start][0 : LEN_OF_OUTPUT]
    to_show_guessed = loaded_matrix_train[start][LEN_OF_OUTPUT : ]
    # print(to_show_wanted.shape)
    # print(to_show_guessed.shape)
    # print(" Target:", to_show_wanted, "\n", "Guess:", to_show_guessed)
    if print_loss:
        loss = loss_function(to_show_guessed, to_show_wanted)
        print(loss)
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
            FL_w[place] = float(to_show_wanted_[i]) * 1700
            FL_g[place] = float(to_show_guessed_[i]) * 1700
            FL_w_[place] = float(to_show_wanted_[i])
            FL_g_[place] = float(to_show_guessed_[i])
        if (i % 5) == 1:
            FR_w[place] = float(to_show_wanted_[i]) * 1700
            FR_g[place] = float(to_show_guessed_[i]) * 1700
            FR_w_[place] = float(to_show_wanted_[i])
            FR_g_[place] = float(to_show_guessed_[i])
        if (i % 5) == 2:    
            RL_w[place] = float(to_show_wanted_[i]) * 1700
            RL_g[place] = float(to_show_guessed_[i]) * 1700
            RL_w_[place] = float(to_show_wanted_[i])
            RL_g_[place] = float(to_show_guessed_[i])
        if (i % 5) == 3:
            RR_w[place] = float(to_show_wanted_[i]) * 1700
            RR_g[place] = float(to_show_guessed_[i]) * 1700
            RR_w_[place] = float(to_show_wanted_[i])
            RR_g_[place] = float(to_show_guessed_[i])
        if (i % 5) == 4:
            SW_w[place] = float(to_show_wanted_[i]) / 60
            SW_g[place] = float(to_show_guessed_[i]) / 60
            SW_w_[place] = float(to_show_wanted_[i])
            SW_g_[place] = float(to_show_guessed_[i])
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
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.plot(FL_w_, label="FL wanted")
    ax1.plot(FL_g_, label="FL guessed")
    ax1.legend(loc=1)
    plt.axis([0, 400, -3.5, 3.5])
    plt.show()
    
    fig = plt.figure()
    ax2 = plt.subplot2grid((1, 1), (0, 0))
    ax2.plot(FR_w_, label="FR wanted")
    ax2.plot(FR_g_, label="FR guessed")
    ax2.legend(loc=1)
    plt.axis([0, 400, -3.5, 3.5])
    plt.show()
    
    fig = plt.figure()
    ax3 = plt.subplot2grid((1, 1), (0, 0))
    ax3.plot(RL_w_, label="RL wanted")
    ax3.plot(RL_g_, label="RL guessed")
    ax3.legend(loc=1)
    plt.axis([0, 400, -3.5, 3.5])
    plt.show()
    
    fig = plt.figure()
    ax4 = plt.subplot2grid((1, 1), (0, 0))
    ax4.plot(RR_w_, label="RR wanted")
    ax4.plot(RR_g_, label="RR guessed")
    ax4.legend(loc=1)
    plt.axis([0, 400, -3.5, 3.5])
    plt.show()
    
    fig = plt.figure()
    ax5 = plt.subplot2grid((1, 1), (0, 0))
    ax5.plot(SW_w_, label="SW wanted")
    ax5.plot(SW_g_, label="SW guessed")
    ax5.legend(loc=1)
    plt.axis([0, 400, -3.5, 3.5])
    plt.show()
    
def plot_losses_after_training(lets_plot=False):
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
    # # losses_pctl = []
    # losses_val_SWl = []
    # losses_val_SW_2l = []
    # losses_val_frontl = []
    # losses_val_rearl = []
    # losses_val_front_brl = []
    # losses_val_rear_brl = []
    # losses_val_front_accl = []
    # losses_val_rear_accl = []
    # segmentsl = []
    # print(len(loaded_matrix_test))
    # print(range(len(loaded_matrix_test)))
    for seg in my_tqdm.tqdm(range(0, len(loaded_matrix_test), 1)):
        #print(seg)
        to_show_wanted = loaded_matrix_test[seg][0 : LEN_OF_OUTPUT]
        to_show_guessed = loaded_matrix_test[seg][LEN_OF_OUTPUT : ]
        if lets_plot:
            lets_see(to_show_wanted, to_show_guessed)
        # diff_pct = 0
        diff_val_SW = 0
        diff_val_SW_2 = 0
        diff_val_front = 0
        diff_val_rear = 0
        diff_val_front_br = 0
        diff_val_front_acc = 0
        diff_val_rear_br = 0
        diff_val_rear_acc = 0
        # diff_val_SWl = 0
        # diff_val_SW_2l = 0
        # diff_val_frontl = 0
        # diff_val_rearl = 0
        # diff_val_front_brl = 0
        # diff_val_front_accl = 0
        # diff_val_rear_brl = 0
        # diff_val_rear_accl = 0
        for moment in range(LEN_OF_OUTPUT):
            if (moment % 5) == 4:
                diff_val_SW += math.pow((float(to_show_guessed [moment]) / 1) - (float(to_show_wanted [moment]) / 1), 2)
                diff_val_SW_2 += math.pow((float(to_show_guessed [moment]) / 1) - (float(to_show_wanted [moment]) / 1), 2)
            if (moment % 5) == 0:
                diff_val_front += math.pow((float(to_show_guessed [moment]) * 1) - (float(to_show_wanted [moment])) * 1, 2)
                if float(to_show_wanted [moment]) < 0:
                    diff_val_front_br += math.pow((float(to_show_guessed [moment]) * 1) - (float(to_show_wanted [moment])) * 1, 2)
                else:
                    diff_val_front_acc += math.pow((float(to_show_guessed [moment]) * 1) - (float(to_show_wanted [moment])) * 1, 2)
            if (moment % 5) == 1:
                diff_val_front += math.pow((float(to_show_guessed [moment]) * 1) - (float(to_show_wanted [moment])) * 1, 2)
                if float(to_show_wanted [moment]) < 0:
                    diff_val_front_br += math.pow((float(to_show_guessed [moment]) * 1) - (float(to_show_wanted [moment])) * 1, 2)
                else:
                    diff_val_front_acc += math.pow((float(to_show_guessed [moment]) * 1) - (float(to_show_wanted [moment])) * 1, 2)
            if (moment % 5) == 2:
                diff_val_rear += math.pow((float(to_show_guessed [moment]) * 1) - (float(to_show_wanted [moment])) * 1, 2)
                if float(to_show_wanted [moment]) < 0:
                    diff_val_rear_br += math.pow((float(to_show_guessed [moment]) * 1) - (float(to_show_wanted [moment])) * 1, 2)
                else:
                    diff_val_rear_acc += math.pow((float(to_show_guessed [moment]) * 1) - (float(to_show_wanted [moment])) * 1, 2)
            if (moment % 5) == 3:
                diff_val_rear += math.pow((float(to_show_guessed [moment]) * 1) - (float(to_show_wanted [moment])) * 1, 2)
                if float(to_show_wanted [moment]) < 0:
                    diff_val_rear_br += math.pow((float(to_show_guessed [moment]) * 1) - (float(to_show_wanted [moment])) * 1, 2)
                else:
                    diff_val_rear_acc += math.pow((float(to_show_guessed [moment]) * 1) - (float(to_show_wanted [moment])) * 1, 2)
            # diff_pct += abs(100  -  ((float(to_show_guessed [moment])) / (float(to_show_wanted [moment])) * 100))
            
            # if (moment % 5) == 4:
            #     diff_val_SWl += loss_function((float(to_show_guessed [moment]) / 1), (float(to_show_wanted [moment]) / 1))
            #     diff_val_SW_2l += loss_function((float(to_show_guessed [moment]) / 1), (float(to_show_wanted [moment]) / 1))
            # if (moment % 5) == 0:
            #     diff_val_frontl += loss_function((float(to_show_guessed [moment]) * 1), (float(to_show_wanted [moment])) * 1)
            #     if float(to_show_wanted [moment]) < 0:
            #         diff_val_front_brl += loss_function((float(to_show_guessed [moment]) * 1), (float(to_show_wanted [moment])) * 1)
            #     else:
            #         diff_val_front_accl += loss_function((float(to_show_guessed [moment]) * 1), (float(to_show_wanted [moment])) * 1)
            # if (moment % 5) == 1:
            #     diff_val_frontl += loss_function((float(to_show_guessed [moment]) * 1), (float(to_show_wanted [moment])) * 1)
            #     if float(to_show_wanted [moment]) < 0:
            #         diff_val_front_brl += loss_function((float(to_show_guessed [moment]) * 1), (float(to_show_wanted [moment])) * 1)
            #     else:
            #         diff_val_front_accl += loss_function((float(to_show_guessed [moment]) * 1), (float(to_show_wanted [moment])) * 1)
            # if (moment % 5) == 2:
            #     diff_val_rearl += loss_function((float(to_show_guessed [moment]) * 1), (float(to_show_wanted [moment])) * 1)
            #     if float(to_show_wanted [moment]) < 0:
            #         diff_val_rear_brl += loss_function((float(to_show_guessed [moment]) * 1), (float(to_show_wanted [moment])) * 1)
            #     else:
            #         diff_val_rear_accl += loss_function((float(to_show_guessed [moment]) * 1), (float(to_show_wanted [moment])) * 1)
            # if (moment % 5) == 3:
            #     diff_val_rearl += loss_function((float(to_show_guessed [moment]) * 1), (float(to_show_wanted [moment])) * 1)
            #     if float(to_show_wanted [moment]) < 0:
            #         diff_val_rear_brl += loss_function((float(to_show_guessed [moment]) * 1), (float(to_show_wanted [moment])) * 1)
            #     else:
            #         diff_val_rear_accl += loss_function((float(to_show_guessed [moment]) * 1), (float(to_show_wanted [moment])) * 1)
            # # diff_pct += abs(100  -  ((float(to_show_guessed [moment])) / (float(to_show_wanted [moment])) * 100))
        
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
        # losses_val_SWl.append(diff_val_SWl / LEN_OF_SEGMENTS)
        # losses_val_SW_2l.append(diff_val_SW_2l / LEN_OF_SEGMENTS)
        # losses_val_frontl.append(diff_val_frontl / LEN_OF_SEGMENTS / 2)
        # losses_val_rearl.append(diff_val_rearl / LEN_OF_SEGMENTS / 2)
        # losses_val_front_brl.append(diff_val_front_brl / LEN_OF_SEGMENTS)
        # losses_val_rear_brl.append(diff_val_rear_brl / LEN_OF_SEGMENTS)
        # losses_val_front_accl.append(diff_val_front_accl / LEN_OF_SEGMENTS)
        # losses_val_rear_accl.append(diff_val_rear_accl / LEN_OF_SEGMENTS)
        # segmentsl.append(seg)
    #print("losses_val_SW [0]: ", losses_val_SW[0])
    #print("losses_val_front [0]: ", losses_val_front[0])
    #print("losses_val_rear [0]: ", losses_val_rear[0])
    
    # print(losses_pct)
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.plot(losses_val_front, 'b', label="MSE per seg in testd WF front wheels")
    ax1.legend(loc=2)
    plt.axis([0, len(loaded_matrix_test), 0, 3])
    plt.show()
    
    fig = plt.figure()
    ax2 = plt.subplot2grid((1, 1), (0, 0))
    ax2.plot(losses_val_rear, 'b', label="MSE per seg in testd WF rear wheels")
    ax2.legend(loc=2)
    plt.axis([0, len(loaded_matrix_test), 0, 3])
    plt.show()
    
    fig = plt.figure()
    ax3 = plt.subplot2grid((1, 1), (0, 0))
    ax3.plot(losses_val_SW, 'b', label="MSE per seg in testd SWA")
    ax3.legend(loc=2)
    plt.axis([0, len(loaded_matrix_test), 0, 3])
    plt.show()
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.plot(losses_val_front_br, 'b', label="MSE per seg in testd WF front wheels while braking")
    ax1.legend(loc=2)
    plt.axis([0, len(loaded_matrix_test), 0, 3])
    plt.show()
    
    fig = plt.figure()
    ax2 = plt.subplot2grid((1, 1), (0, 0))
    ax2.plot(losses_val_front_acc, 'b', label="MSE per seg in testd WF front wheels while accelerating")
    ax2.legend(loc=2)
    plt.axis([0, len(loaded_matrix_test), 0, 3])
    plt.show()
    
    fig = plt.figure()
    ax3 = plt.subplot2grid((1, 1), (0, 0))
    ax3.plot(losses_val_rear_br, 'b', label="MSE per seg in testd WF rear wheels while braking")
    ax3.legend(loc=2)
    plt.axis([0, len(loaded_matrix_test), 0, 3])
    plt.show()
    
    fig = plt.figure()
    ax4 = plt.subplot2grid((1, 1), (0, 0))
    ax4.plot(losses_val_rear_acc, 'b', label="MSE per seg in testd WF rear wheels while accelerating")
    ax4.legend(loc=2)
    plt.axis([0, len(loaded_matrix_test), 0, 3])
    plt.show()
    
    fig = plt.figure()
    ax5 = plt.subplot2grid((1, 1), (0, 0))
    ax5.plot(losses_val_SW_2, 'b', label="MSE per seg in testd SW")
    ax5.legend(loc=2)
    plt.axis([0, len(loaded_matrix_test), 0, 3])
    plt.show()
    
    # fig = plt.figure()
    # ax1 = plt.subplot2grid((1, 1), (0, 0))
    # ax1.plot(losses_val_frontl, 'b', label="MSE per seg in testd WF front wheels")
    # ax1.legend(loc=2)
    # plt.axis([0, len(loaded_matrix_test), 0, 3])
    # plt.show()
    
    # fig = plt.figure()
    # ax2 = plt.subplot2grid((1, 1), (0, 0))
    # ax2.plot(losses_val_rearl, 'b', label="MSE per seg in testd WF rear wheels")
    # ax2.legend(loc=2)
    # plt.axis([0, len(loaded_matrix_test), 0, 3])
    # plt.show()
    
    # fig = plt.figure()
    # ax3 = plt.subplot2grid((1, 1), (0, 0))
    # ax3.plot(losses_val_SWl, 'b', label="MSE per seg in testd SWA")
    # ax3.legend(loc=2)
    # plt.axis([0, len(loaded_matrix_test), 0, 3])
    # plt.show()
    
    # fig = plt.figure()
    # ax1 = plt.subplot2grid((1, 1), (0, 0))
    # ax1.plot(losses_val_front_brl, 'b', label="MSE per seg in testd WF front wheels while braking")
    # ax1.legend(loc=2)
    # plt.axis([0, len(loaded_matrix_test), 0, 3])
    # plt.show()
    
    # fig = plt.figure()
    # ax2 = plt.subplot2grid((1, 1), (0, 0))
    # ax2.plot(losses_val_front_accl, 'b', label="MSE per seg in testd WF front wheels while accelerating")
    # ax2.legend(loc=2)
    # plt.axis([0, len(loaded_matrix_test), 0, 3])
    # plt.show()
    
    # fig = plt.figure()
    # ax3 = plt.subplot2grid((1, 1), (0, 0))
    # ax3.plot(losses_val_rear_brl, 'b', label="MSE per seg in testd WF rear wheels while braking")
    # ax3.legend(loc=2)
    # plt.axis([0, len(loaded_matrix_test), 0, 3])
    # plt.show()
    
    # fig = plt.figure()
    # ax4 = plt.subplot2grid((1, 1), (0, 0))
    # ax4.plot(losses_val_rear_accl, 'b', label="MSE per seg in testd WF rear wheels while accelerating")
    # ax4.legend(loc=2)
    # plt.axis([0, len(loaded_matrix_test), 0, 3])
    # plt.show()
    
    # fig = plt.figure()
    # ax5 = plt.subplot2grid((1, 1), (0, 0))
    # ax5.plot(losses_val_SW_2l, 'b', label="MSE per seg in testd SW")
    # ax5.legend(loc=2)
    # plt.axis([0, len(loaded_matrix_test), 0, 3])
    # plt.show()

plot_losses_after_training(True)
