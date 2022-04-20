# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:38:30 2022

@author: Bagoly Zoltán
"""
import math
import numpy as np
import tqdm as my_tqdm
import matplotlib.pyplot as plt
from matplotlib import style
import torch.nn as nn
LEN_OF_SEGMENTS = 400
LEN_OF_INPUT = 4 * LEN_OF_SEGMENTS
LEN_OF_OUTPUT = 5 * LEN_OF_SEGMENTS

# adat elokeszites
NETS = np.array([2, 4, 6, 8, 10])      # numbers of layers
NUM_OF_TESTS = 68
EPOCHS = 400
SCALED = True

loaded_losses = np.zeros([len(NETS), EPOCHS, 3], dtype=float)
loaded_matrices = np.zeros([len(NETS), EPOCHS, NUM_OF_TESTS, 2, int(LEN_OF_OUTPUT / LEN_OF_SEGMENTS), LEN_OF_SEGMENTS], dtype=float)
loss_function = nn.MSELoss()

# retrieving data from file
def load_data():
    i = 0
    for net in NETS:
        loaded_losses[i] = np.load('losses_and_val_losses_of_{}.npy'.format(net))
        loaded_matrices[i] = np.load('matrices_of_{}.npy'.format(net))
        i += 1
load_data()


style.use("ggplot")

# model_name = TRAINING_NAME
def loss_graph():
    i = 0
    for net in NETS:
        plt.title(str(net))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(loaded_losses[i, :, 1], 'orange', label="loss")
        plt.plot(loaded_losses[i, :, 2], 'blue', label="val loss")
        plt.legend(loc=1)
        plt.axis([-5, 400, 0, 2])
        plt.show()
        i += 1
    
loss_graph()

def one_segment_test(net=4, epoch=399, test_seg=67, print_loss=False):
    to_show_wanted = loaded_matrices[net, epoch, test_seg, 0]
    to_show_guessed = loaded_matrices[net, epoch, test_seg, 1]
    if print_loss:
        loss = loss_function(to_show_guessed, to_show_wanted)
        print(loss)
    
    ax1 = plt.subplot2grid((5, 1), (0, 0))
    ax2 = plt.subplot2grid((5, 1), (1, 0))
    ax3 = plt.subplot2grid((5, 1), (2, 0))
    ax4 = plt.subplot2grid((5, 1), (3, 0))
    ax5 = plt.subplot2grid((5, 1), (4, 0))
                           
    ax1.plot(to_show_wanted [0, :] *1700, label="FL wan")
    ax1.plot(to_show_guessed[0, :] *1700, label="FL gue")
    ax1.legend(loc=1)
    
    ax2.plot(to_show_wanted [1, :] *1700, label="FR wan")
    ax2.plot(to_show_guessed[1, :] *1700, label="FR gue")
    ax2.legend(loc=1)
    
    ax3.plot(to_show_wanted [2, :] *1700, label="RL wan")
    ax3.plot(to_show_guessed[2, :] *1700, label="RL gue")
    ax3.legend(loc=1)
    
    ax4.plot(to_show_wanted [3, :] *1700, label="RR wan")
    ax4.plot(to_show_guessed[3, :] *1700, label="RR gue")
    ax4.legend(loc=1)
    
    ax5.plot(to_show_wanted [4, :] /60, label="SW wan")
    ax5.plot(to_show_guessed[4, :] /60, label="SW gue")
    ax5.legend(loc=1)
    
    plt.show()
    
    plt.title("Front left")
    plt.xlabel('moment')
    plt.ylabel('raw output')
    plt.plot(to_show_wanted [0, :], label="wan")
    plt.plot(to_show_guessed[0, :], label="gue")
    plt.legend(loc=1)
    plt.axis([0, 400, -3.5, 3.5])
    plt.show()
    
    plt.title("Front right")
    plt.xlabel('moment')
    plt.ylabel('raw output')
    plt.plot(to_show_wanted [1, :], label="wan")
    plt.plot(to_show_guessed[1, :], label="gue")
    plt.legend(loc=1)
    plt.axis([0, 400, -3.5, 3.5])
    plt.show()
    
    plt.title("Rear left")
    plt.xlabel('moment')
    plt.ylabel('raw output')
    plt.plot(to_show_wanted [2, :], label="wan")
    plt.plot(to_show_guessed[2, :], label="gue")
    plt.legend(loc=1)
    plt.axis([0, 400, -3.5, 3.5])
    plt.show()
    
    plt.title("Rear right")
    plt.xlabel('moment')
    plt.ylabel('raw output')
    plt.plot(to_show_wanted [3, :], label="wan")
    plt.plot(to_show_guessed[3, :], label="gue")
    plt.legend(loc=1)
    plt.axis([0, 400, -3.5, 3.5])
    plt.show()
    
    plt.title("Steering wheel")
    plt.xlabel('moment')
    plt.ylabel('raw output')
    plt.plot(to_show_wanted [4, :], label="wan")
    plt.plot(to_show_guessed[4, :], label="gue")
    plt.legend(loc=1)
    plt.axis([0, 400, -3.5, 3.5])
    plt.show()
    
    return to_show_wanted, to_show_guessed
wanted, guessed = one_segment_test()

def plot_losses_separated(net, epoch, lets_plot=True):
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

    for seg in my_tqdm.tqdm(range(0, len(loaded_matrices[0][0]), 1)):
        #print(seg)
        to_show_wanted = loaded_matrices[net, epoch, seg, 0]
        to_show_guessed = loaded_matrices[net, epoch, seg, 1]

        diff_val_SW = 0
        diff_val_SW_2 = 0
        diff_val_front = 0
        diff_val_rear = 0
        diff_val_front_br = 0
        diff_val_front_acc = 0
        diff_val_rear_br = 0
        diff_val_rear_acc = 0

        for data_type in range(int(LEN_OF_OUTPUT / LEN_OF_SEGMENTS)):
            for moment in range(LEN_OF_SEGMENTS):
                if (moment % 5) == 4:
                    diff_val_SW += math.pow((float(to_show_guessed [data_type][moment]) / 1) - (float(to_show_wanted [data_type][moment]) / 1), 2)
                    diff_val_SW_2 += math.pow((float(to_show_guessed [data_type][moment]) / 1) - (float(to_show_wanted [data_type][moment]) / 1), 2)
                if (moment % 5) == 0:
                    diff_val_front += math.pow((float(to_show_guessed [data_type][moment]) * 1) - (float(to_show_wanted [data_type][moment])) * 1, 2)
                    if float(to_show_wanted [data_type][moment]) < 0:
                        diff_val_front_br += math.pow((float(to_show_guessed [data_type][moment]) * 1) - (float(to_show_wanted [data_type][moment])) * 1, 2)
                    else:
                        diff_val_front_acc += math.pow((float(to_show_guessed [data_type][moment]) * 1) - (float(to_show_wanted [data_type][moment])) * 1, 2)
                if (moment % 5) == 1:
                    diff_val_front += math.pow((float(to_show_guessed[data_type] [moment]) * 1) - (float(to_show_wanted [data_type][moment])) * 1, 2)
                    if float(to_show_wanted [data_type][moment]) < 0:
                        diff_val_front_br += math.pow((float(to_show_guessed [data_type][moment]) * 1) - (float(to_show_wanted [data_type][moment])) * 1, 2)
                    else:
                        diff_val_front_acc += math.pow((float(to_show_guessed [data_type][moment]) * 1) - (float(to_show_wanted [data_type][moment])) * 1, 2)
                if (moment % 5) == 2:
                    diff_val_rear += math.pow((float(to_show_guessed [data_type][moment]) * 1) - (float(to_show_wanted [data_type][moment])) * 1, 2)
                    if float(to_show_wanted [data_type][moment]) < 0:
                        diff_val_rear_br += math.pow((float(to_show_guessed [data_type][moment]) * 1) - (float(to_show_wanted [data_type][moment])) * 1, 2)
                    else:
                        diff_val_rear_acc += math.pow((float(to_show_guessed [data_type][moment]) * 1) - (float(to_show_wanted [data_type][moment])) * 1, 2)
                if (moment % 5) == 3:
                    diff_val_rear += math.pow((float(to_show_guessed [data_type][moment]) * 1) - (float(to_show_wanted [data_type][moment])) * 1, 2)
                    if float(to_show_wanted [data_type][moment]) < 0:
                        diff_val_rear_br += math.pow((float(to_show_guessed [data_type][moment]) * 1) - (float(to_show_wanted [data_type][moment])) * 1, 2)
                    else:
                        diff_val_rear_acc += math.pow((float(to_show_guessed [data_type][moment]) * 1) - (float(to_show_wanted [data_type][moment])) * 1, 2)
        
        losses_val_SW.append(diff_val_SW / LEN_OF_SEGMENTS)
        losses_val_SW_2.append(diff_val_SW_2 / LEN_OF_SEGMENTS)
        losses_val_front.append(diff_val_front / LEN_OF_SEGMENTS / 2)
        losses_val_rear.append(diff_val_rear / LEN_OF_SEGMENTS / 2)
        losses_val_front_br.append(diff_val_front_br / LEN_OF_SEGMENTS)
        losses_val_rear_br.append(diff_val_rear_br / LEN_OF_SEGMENTS)
        losses_val_front_acc.append(diff_val_front_acc / LEN_OF_SEGMENTS)
        losses_val_rear_acc.append(diff_val_rear_acc / LEN_OF_SEGMENTS)
        segments.append(seg)
    
    plt.title("MSE per seg in testd WF front wheels")
    plt.xlabel('test segment')
    plt.ylabel('loss')
    plt.plot(losses_val_front)
    plt.legend(loc=2)
    plt.axis([0, len(loaded_matrices[0][0]), 0, 3])
    plt.show()
    
    plt.title("MSE per seg in testd WF rear wheels")
    plt.xlabel('test segment')
    plt.ylabel('loss')
    plt.plot(losses_val_rear)
    plt.legend(loc=2)
    plt.axis([0, len(loaded_matrices[0][0]), 0, 3])
    plt.show()
    
    plt.title("MSE per seg in testd SWA")
    plt.xlabel('test segment')
    plt.ylabel('loss')
    plt.plot(losses_val_SW)
    plt.legend(loc=2)
    plt.axis([0, len(loaded_matrices[0][0]), 0, 3])
    plt.show()
    
    plt.title("MSE per seg in testd WF front wheels while braking")
    plt.xlabel('test segment')
    plt.ylabel('loss')
    plt.plot(losses_val_front_br)
    plt.legend(loc=2)
    plt.axis([0, len(loaded_matrices[0][0]), 0, 3])
    plt.show()
    
    plt.title("MSE per seg in testd WF front wheels while accelerating")
    plt.xlabel('test segment')
    plt.ylabel('loss')
    plt.plot(losses_val_front_acc)
    plt.legend(loc=2)
    plt.axis([0, len(loaded_matrices[0][0]), 0, 3])
    plt.show()
    
    plt.title("MSE per seg in testd WF rear wheels while braking")
    plt.xlabel('test segment')
    plt.ylabel('loss')
    plt.plot(losses_val_rear_br)
    plt.legend(loc=2)
    plt.axis([0, len(loaded_matrices[0][0]), 0, 3])
    plt.show()
    
    plt.title("MSE per seg in testd WF rear wheels while accelerating")
    plt.xlabel('test segment')
    plt.ylabel('loss')
    plt.plot(losses_val_rear_acc)
    plt.legend(loc=2)
    plt.axis([0, len(loaded_matrices[0][0]), 0, 3])
    plt.show()
    
    plt.title("MSE per seg in testd SW")
    plt.xlabel('test segment')
    plt.ylabel('loss')
    plt.plot(losses_val_SW_2)
    plt.legend(loc=2)
    plt.axis([0, len(loaded_matrices[0][0]), 0, 3])
    plt.show()

plot_losses_separated(0, 399, True)
