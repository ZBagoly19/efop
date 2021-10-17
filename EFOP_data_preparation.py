"""
@author: Bagoly Zoltán
         zoltan.bagoly@gmail.com
"""
import random
import scipy.io
import math
import numpy as np


# adat elokeszites
my_training_data = []
my_testing_data = []
LEN_OF_SEGMENTS = 400
LEN_OF_INPUT = 4 * LEN_OF_SEGMENTS
LEN_OF_OUTPUT = 5 * LEN_OF_SEGMENTS
DATA_STRIDE = 10
# mat = scipy.io.loadmat("meas01.mat")

class Data_read():
        
    def read_from_raw_data(self, source, start_of_last):
        print("Data preparation")
        mat = scipy.io.loadmat(source)
        
        # Kimenetek
        Fy_FL = mat["WheForceX_FL"]
        Fy_FR = mat["WheForceX_FR"]
        Fy_RL = mat["WheForceX_RL"]
        Fy_RR = mat["WheForceX_RR"]
        WheelAngle = mat["WheAng"]
        
        # Bemenetek
        Orientation = mat["Ori_Z"]
        Pos_X = mat["Pos_X"]
        Pos_Y = mat["Pos_Y"]
        Velocity = mat["Vel"]
        
        segment = 0
        while segment <= start_of_last:
        # for segment in range(0, start_of_last, DATA_STRIDE):
            # print("segment", segment)
            target = my_training_data
            my_input = []
            my_output = []
            
            angle = -1 * Orientation [segment]
            
            rand_x = random.randint(-100, 100)
            rand_y = random.randint(-100, 100)
            rand_x = 0
            rand_y = 0
            
            for i in range(LEN_OF_SEGMENTS):
               # print("i", i)
                # print("segment + i", segment + i)
                
                # X es Y koordinatak szegmensenkent 0-ba tolasa, iranyba forgatasa
                # print("angle:", angle)
                x_norm = Pos_X [segment + i] - (Pos_X [segment])
                y_norm = Pos_Y [segment + i] - (Pos_Y [segment])
                # print("X norm:", x_norm, "Y norm:", y_norm)
                x_norm_rot = math.cos(angle) * (x_norm) - math.sin(angle) * (y_norm)
                y_norm_rot = math.sin(angle) * (x_norm) + math.cos(angle) * (y_norm)
                # print("X norm rot:", x_norm_rot, "Y norm rot:", y_norm_rot)
                
                # normalise between -3 and 3
                my_input.append((x_norm_rot / 100) + rand_x)
                my_input.append((y_norm_rot / 100) + rand_y)
                my_input.append(Velocity [segment] / 10)
                my_input.append(Orientation [segment])
                
                my_output.append(Fy_FL [segment + i] / 1700)
                my_output.append(Fy_FR [segment + i] / 1700)
                my_output.append(Fy_RL [segment + i] / 1700)
                my_output.append(Fy_RR [segment + i] / 1700)
                my_output.append(WheelAngle [segment + i] * 60)
            
            chance_of_test_data = random.randint(0, 10)
            # print(chance_of_test_data, chance_of_test_data == 1)
            if chance_of_test_data == 1:
                target = my_testing_data
                segment += LEN_OF_SEGMENTS - DATA_STRIDE
            
            target.append([np.array(my_input), np.array(my_output)])
            # print(segment, "my_out", np.array(my_output), np.array(my_output).shape)
            # print("my_in", np.array(my_output).shape, np.array(my_input))
            
            segment += DATA_STRIDE
            
        print("my_testing_data", np.array(my_testing_data).shape)
        print("train_data", np.array(my_training_data).shape)
        np.random.shuffle(my_testing_data)
        np.save("my_testing_data.npy", my_testing_data)
        np.random.shuffle(my_training_data)
        np.save("my_training_data.npy", my_training_data)
        
dr = Data_read()
dr.read_from_raw_data("meas01.mat", 10118 - LEN_OF_SEGMENTS)
dr.read_from_raw_data("meas02.mat", 10118 - LEN_OF_SEGMENTS)