"""
@author: Bagoly Zoltán
         zoltan.bagoly@gmail.com
"""
import random
import scipy.io
import math
import numpy as np
import time
import matplotlib.pyplot as plt


# adat elokeszites
my_training_data = []
my_testing_data = []
LEN_OF_SEGMENTS = 100
LEN_OF_INPUT = 3 * LEN_OF_SEGMENTS
LEN_OF_OUTPUT = 2 * LEN_OF_SEGMENTS
DATA_STRIDE = 10
TEST_DATA_PART_RECIP = 5
TESTED_BIT = 2             # kisebb kell legyen a TEST_DATA_PART_RECIP-nel
time_ = int(time.time())
print("time:", time_)
glob_train_segs = []
glob_test_segs = []
x_glob_train_tmp = []
y_glob_train_tmp = []
x_glob_test_tmp = []
y_glob_test_tmp = []
# mat = scipy.io.loadmat("meas01.mat")

class Data_read():
        
    def read_from_raw_data(self, source, start_of_last):
        print("Data preparation")
        mat = scipy.io.loadmat(source)
        
        # Kimenetek
        AngVel = mat["a__AngVel_Z"]
        Vel = mat["a__Vel"]
        
        # Bemenetek
        Orientation = mat["a__Ori_Z"]
        Pos_X = mat["a__Pos_X"]
        Pos_Y = mat["a__Pos_Y"]
        
        segment = 0
        while segment <= start_of_last:
        # for segment in range(0, start_of_last, DATA_STRIDE):
            # print("segment", segment)
            target = my_training_data
            trg = "train"
            # if we will stay in bounds if we get a test data
            if segment + (LEN_OF_SEGMENTS - DATA_STRIDE) + LEN_OF_SEGMENTS <= start_of_last:
                if segment % (LEN_OF_SEGMENTS * TEST_DATA_PART_RECIP) == LEN_OF_SEGMENTS * TESTED_BIT:
                    target = my_testing_data
                    trg = "test"
                    segment += LEN_OF_SEGMENTS - DATA_STRIDE
                else:
                    target = my_training_data
                    trg = "train"
            else:
                target = my_training_data
                trg = "train"
                
            my_input = [None] * LEN_OF_INPUT
            my_output = [None] * LEN_OF_OUTPUT
            
            angle = -1 * Orientation [segment]
            
            rand_x = random.randint(-100, 100)
            rand_y = random.randint(-100, 100)
            rand_x = 0
            rand_y = 0
            
            x_glob_test_tmp = [None] * LEN_OF_SEGMENTS
            y_glob_test_tmp = [None] * LEN_OF_SEGMENTS
            x_glob_train_tmp = [None] * LEN_OF_SEGMENTS
            y_glob_train_tmp = [None] * LEN_OF_SEGMENTS
            for i in range(LEN_OF_SEGMENTS):
               # print("i", i)
                # print("segment + i", segment + i)
                
                if trg == "test":
                    x_glob_test_tmp[i] = Pos_X [segment + i]
                    y_glob_test_tmp[i] = Pos_Y [segment + i]
                else:
                    x_glob_train_tmp[i] = Pos_X [segment + i]
                    y_glob_train_tmp[i] = Pos_Y [segment + i]
                
                # X es Y koordinatak szegmensenkent 0-ba tolasa, iranyba forgatasa
                # print("angle:", angle)
                x_norm = Pos_X [segment + i] - (Pos_X [segment])
                y_norm = Pos_Y [segment + i] - (Pos_Y [segment])
                # print("X norm:", x_norm, "Y norm:", y_norm)
                x_norm_rot = math.cos(angle) * (x_norm) - math.sin(angle) * (y_norm)
                y_norm_rot = math.sin(angle) * (x_norm) + math.cos(angle) * (y_norm)
                # print("X norm rot:", x_norm_rot, "Y norm rot:", y_norm_rot)
                
                # normalise between -3 and 3
                my_input[(i * int(LEN_OF_INPUT / LEN_OF_SEGMENTS)) + 0] = (x_norm_rot / 30) + rand_x #30, 100
                my_input[(i * int(LEN_OF_INPUT / LEN_OF_SEGMENTS)) + 1] = (y_norm_rot / 30) + rand_y
                my_input[(i * int(LEN_OF_INPUT / LEN_OF_SEGMENTS)) + 2] = Orientation [segment + i] / 2 #30, 1
                # my_input[i] = [(x_norm_rot / 30) + rand_x, (y_norm_rot / 30) + rand_y, 
                #                   Velocity [segment] / 100, Orientation [segment] / 30]
                # my_input[i] = [(x_norm_rot / 1) + rand_x, (y_norm_rot / 1) + rand_y, 
                #                   Velocity [segment] / 1, Orientation [segment] / 1]
                
                my_output[(i * int(LEN_OF_OUTPUT / LEN_OF_SEGMENTS)) + 0] = Vel [segment + i] / 10
                my_output[(i * int(LEN_OF_OUTPUT / LEN_OF_SEGMENTS)) + 1] = AngVel [segment + i] * 6
                # # my_output[i] = [Fy_FL [segment + i] / 1700, Fy_FR [segment + i] / 1700,
                #                   Fy_RL [segment + i] / 1700, Fy_RR [segment + i] / 1700, 
                #                   WheelAngle [segment + i] * 60]
                
            
            # if len(my_input) != 400:
            #     print(len(my_input))
            # A [:] nelkul csak referencia atadas tortenik, .clear()-kor torli innen is
            if trg == "test":
                segment += LEN_OF_SEGMENTS - DATA_STRIDE
                glob_test_segs.append([x_glob_test_tmp[:], y_glob_test_tmp[:]])
            else:
                glob_train_segs.append([x_glob_train_tmp[:], y_glob_train_tmp[:]])
            
            target.append([np.array(my_input), np.array(my_output)])
            #print(segment, "my_out", np.array(my_output), np.array(my_output).shape)
            #print("my_in", np.array(my_output).shape, np.array(my_input))
            
            segment += DATA_STRIDE
        
        print("done, saving")
        #print("test data", np.array(my_testing_data).shape)
        #print("train data", np.array(my_training_data).shape)
        np.random.shuffle(my_testing_data)
        np.save("my_testing_data_XYO_VAV_better_skale.npy", my_testing_data)
        np.random.shuffle(my_training_data)
        np.save("my_training_data_XYO_VAV_better_skale.npy", my_training_data)
        
dr = Data_read()
#dr.read_from_raw_data("meas01.mat", 10118 - LEN_OF_SEGMENTS)
#dr.read_from_raw_data("meas02.mat", 10305 - LEN_OF_SEGMENTS)
#dr.read_from_raw_data("data_zoli_carmaker_leaf_02.mat", 137238 - LEN_OF_SEGMENTS)
dr.read_from_raw_data("meas_X_Y_Ori_Vel_AngVel.mat", 13724 - LEN_OF_SEGMENTS)


# Vizualizacio
# print("done, visualizing")
# def visualization(only_tests=False, only_one=False, which_one=0):
#     ax0 = plt.subplot2grid((1, 1), (0, 0))
#     for segm in glob_test_segs:
#         ax0.plot(segm[0], segm[1])
#     if only_tests == False:
#         for segm in glob_train_segs:
#             ax0.plot(segm[0], segm[1])
#     plt.show()
    
#     if only_tests == False:
#         if only_one:
#             x_norm_ = []
#             y_norm_ = []
#             for i in range(my_testing_data[0][0].size):
#                 if i % int(LEN_OF_INPUT / LEN_OF_SEGMENTS) == 0:
#                     x_norm_.append(my_testing_data[which_one][0][i])
#                 if i % int(LEN_OF_INPUT / LEN_OF_SEGMENTS) == 1:
#                     y_norm_.append(my_testing_data[which_one][0][i])
#             plt.plot(x_norm_, y_norm_)
#             plt.axis([-0, 10, -5, 5])
#             plt.show()
#         else:
#             for seg in my_testing_data:
#                 x_norm_ = [] 
#                 y_norm_ = []
#                 for i in range(my_testing_data[1][0].size):
#                     if i % int(LEN_OF_INPUT / LEN_OF_SEGMENTS) == 0:
#                         x_norm_.append(seg[0][i])
#                     if i % int(LEN_OF_INPUT / LEN_OF_SEGMENTS) == 1:
#                         y_norm_.append(seg[0][i])
#                 plt.plot(x_norm_, y_norm_)
#                 plt.axis([-0, 10, -5, 5])
#                 plt.show()
#
# visualization()
# visualization(only_one=True)
# visualization(only_tests=True)
