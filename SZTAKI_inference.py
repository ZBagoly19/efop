"""
@author: Bagoly Zoltán
"""
import os
import math
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
    # ezekkel adjuk meg, mely háló fájt töltjük be
INIT_LRS = [0.0400, 0.0200, 0.0150, 0.0100, 0.0050, 0.0020, 0.0005]
learn_rate = INIT_LRS[0]
FILE_PRE = "10_sched_test_Adam_" + str(learn_rate) + "_sched_39_0.7_bn_no_reg_no" + "_"
FILE_NAME_START = 'XYO_VAV__'
WIDTH_OF_LAYERS = 24
NUM_OF_LAYERS = 2
EPOCH = 599
create_matrices = True    # ez csak a konkrét szegmensek vizsgálásához kell

##############################################################################
# ehhez nem kell nyúlni

if consts.EPOCHS <= EPOCH:
    print("Túl magas a epoch érték!")

# GPU
def on_gpu():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

colours = ['black', 'blue', 'green', 'brown', 'red', 'cyan', 'magenta', 
           'yellow', 'darkblue', 'orange', 'pink', 'beige', 'coral', 'crimson', 
           'darkgreen', 'fuchsia', 'goldenrod', 'grey', 'yellowgreen', 'lavender', 
           'lightblue', 'lime', 'navy', 'sienna', 'silver',
           'orchid', 'wheat', 'white', 'chocolate', 'khaki', 'azure',
           'salmon', 'plum']
styles = ['solid', 'dotted', 'dashed']
styles = ['-', '--', '-.', ':', 'solid']

DEVICE = on_gpu()

loss_function = nn.MSELoss()

if create_matrices:
    matrix_loaded_losses = np.load(FILE_PRE + FILE_NAME_START + 'losses_and_val_losses_of_{}_{}.npy'.format(WIDTH_OF_LAYERS, NUM_OF_LAYERS))
    
    LEN_OF_INPUT = consts.NUM_OF_INPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS
    LEN_OF_OUTPUT = int(consts.NUM_OF_OUTPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS * consts.INPUT_OUTPUT_COEFFICIENT)
    
    SHUF_TEST_DATA = np.load("my_testing_data_XYO_VAV_better_skale_mom1sec.npy", allow_pickle=True)
    SHUF_TRAIN_DATA = np.load("my_training_data_XYO_VAV_better_skale_mom1sec.npy", allow_pickle=True)
    TRAIN_X = torch.Tensor([i[0] for i in SHUF_TRAIN_DATA]).view(-1, LEN_OF_INPUT)
    TEST_X = torch.Tensor([i[0] for i in SHUF_TEST_DATA]).view(-1, LEN_OF_INPUT)
    TRAIN_Y = torch.Tensor([i[1] for i in SHUF_TRAIN_DATA]).view(-1, LEN_OF_OUTPUT)
    TEST_Y = torch.Tensor([i[1] for i in SHUF_TEST_DATA]).view(-1, LEN_OF_OUTPUT)
    
    matrix_test_glob = np.zeros([len(TEST_X), 2, consts.NUM_OF_OUTPUT_DATA_TYPES, consts.LEN_OF_SEGMENTS], dtype=float)
    matrix_train_glob = np.zeros([len(TRAIN_X), 2, consts.NUM_OF_OUTPUT_DATA_TYPES, consts.LEN_OF_SEGMENTS], dtype=float)
    
    num_of_layers = NUM_OF_LAYERS
    
    # Net fully connected
    class Net_fc(nn.Module):
        def __init__(self):
            super().__init__()
            
            x = torch.randn(LEN_OF_INPUT).view(-1, 1, LEN_OF_INPUT)
            self._to_linear = None
            self.linearize(x)
            
            self.fc1 = nn.Linear(self._to_linear, WIDTH_OF_LAYERS)
            #nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
            if 1 < num_of_layers:
                self.fc2 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                #nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
            if 2 < num_of_layers:
                self.fc3 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                #nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
                self.fc4 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                #nn.init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
            if 4 < num_of_layers:
                self.fc5 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                #nn.init.kaiming_uniform_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
                self.fc6 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                #nn.init.kaiming_uniform_(self.fc6.weight, mode='fan_in', nonlinearity='relu')
            if 6 < num_of_layers:
                self.fc7 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                #nn.init.kaiming_uniform_(self.fc7.weight, mode='fan_in', nonlinearity='relu')
                self.fc8 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                #nn.init.kaiming_uniform_(self.fc8.weight, mode='fan_in', nonlinearity='relu')
            if 8 < num_of_layers:
                self.fc9 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                #nn.init.kaiming_uniform_(self.fc9.weight, mode='fan_in', nonlinearity='relu')
                self.fc10 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                #nn.init.kaiming_uniform_(self.fc10.weight, mode='fan_in', nonlinearity='relu')
            if 10 < num_of_layers:
                self.fc11 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                #nn.init.kaiming_uniform_(self.fc11.weight, mode='fan_in', nonlinearity='relu')
                self.fc12 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                #nn.init.kaiming_uniform_(self.fc12.weight, mode='fan_in', nonlinearity='relu')
            if 12 < num_of_layers:
                self.fc13 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                #nn.init.kaiming_uniform_(self.fc13.weight, mode='fan_in', nonlinearity='relu')
                self.fc14 = nn.Linear(WIDTH_OF_LAYERS, WIDTH_OF_LAYERS)
                #nn.init.kaiming_uniform_(self.fc14.weight, mode='fan_in', nonlinearity='relu')
            self.fc_last = nn.Linear(WIDTH_OF_LAYERS, LEN_OF_OUTPUT)
            #nn.init.kaiming_uniform_(self.fc_last.weight, mode='fan_in', nonlinearity='relu')
            
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
            # print("last", x.shape, x[0], x)
            
            return x
    
    f = FILE_PRE + 'net_' + FILE_NAME_START + '{}_{}_{}.pth'.format(WIDTH_OF_LAYERS, NUM_OF_LAYERS, EPOCH)
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
            for seg in range(len(TEST_X)):
                wanted, guessed = one_segment_test(seg)
                a = wanted.cpu()
                b = guessed.cpu()
                c = a.detach().numpy()
                d = b.detach().numpy()
                matrix_row = np.column_stack([c, d])
                matrix_test[seg] = matrix_row
            matrix_test_l_formed = np.zeros([len(TEST_X), 2, consts.NUM_OF_OUTPUT_DATA_TYPES, int(consts.LEN_OF_SEGMENTS * consts.INPUT_OUTPUT_COEFFICIENT)], dtype=float)
            for test_case in range(len(TEST_X)):
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
            matrix_train_l_formed = np.zeros([len(TRAIN_X), 2, consts.NUM_OF_OUTPUT_DATA_TYPES, int(consts.LEN_OF_SEGMENTS * consts.INPUT_OUTPUT_COEFFICIENT)], dtype=float)
            for seg in range(len(TRAIN_X)):
                wanted, guessed = one_segment_test_on_train(seg)
                a = wanted.cpu()
                b = guessed.cpu()
                c = a.detach().numpy()
                d = b.detach().numpy()
                matrix_row = np.column_stack([c, d])
                matrix_train[seg] = matrix_row
            for train_case in range(len(TRAIN_X)):
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
    else:
        print("Sikertelen háló betöltés!")
    
def segments_test(figure_const, test_segments_l, train_segments_l):
    for test_seg in test_segments_l:
        if len(matrix_test_glob) <= test_seg:
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
        plt.axis([0, int(consts.LEN_OF_SEGMENTS * consts.INPUT_OUTPUT_COEFFICIENT) - 1, -3.5, 3.5])
        plt.show()
        
        plt.figure(100000+figure_const*10000+test_seg*10+2)
        plt.title("Angular Velocity")
        plt.xlabel('moment')
        plt.ylabel('raw output')
        plt.plot(to_show_wanted [1, :], label="wan")
        plt.plot(to_show_guessed[1, :], label="gue")
        plt.legend(loc=1)
        plt.axis([0, int(consts.LEN_OF_SEGMENTS * consts.INPUT_OUTPUT_COEFFICIENT) - 1, -3.5, 3.5])
        plt.show()
        
        plt.show()
        
    for train_seg in train_segments_l:
        if len(matrix_train_glob) <= train_seg:
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
        plt.axis([0, int(consts.LEN_OF_SEGMENTS * consts.INPUT_OUTPUT_COEFFICIENT) - 1, -3.5, 3.5])
        plt.show()
        
        plt.figure(200000+figure_const*10000+train_seg*100+2)
        plt.title("Angular Velocity")
        plt.xlabel('moment')
        plt.ylabel('raw output')
        plt.plot(to_show_wanted [1, :], label="wan")
        plt.plot(to_show_guessed[1, :], label="gue")
        plt.legend(loc=1)
        plt.axis([0, int(consts.LEN_OF_SEGMENTS * consts.INPUT_OUTPUT_COEFFICIENT) - 1, -3.5, 3.5])
        plt.show()
        
        plt.show()

def loss_check(figure_const, segments_l, train=False):
    losses_test_Vel = np.zeros([len(segments_l)], dtype=float)
    losses_test_AngVel = np.zeros([len(segments_l)], dtype=float)
    losses_per_test_segment = np.zeros([len(segments_l)], dtype=float)
    
    idx = 0
    for seg in segments_l:
        #print(seg)
        if train:
            to_show_wanted = matrix_train_glob[seg, 0]
            to_show_guessed = matrix_train_glob[seg, 1]
        else:
            to_show_wanted = matrix_test_glob[seg, 0]
            to_show_guessed = matrix_test_glob[seg, 1]
        tensor_w = torch.from_numpy(to_show_wanted)
        tensor_g = torch.from_numpy(to_show_guessed)                
    
        diff_test_Vel = 0
        diff_test_AngVel = 0
    
        for moment in range(int(consts.LEN_OF_SEGMENTS * consts.INPUT_OUTPUT_COEFFICIENT)):
            diff_test_Vel += math.pow((float(to_show_guessed [0, moment])) - (float(to_show_wanted [0, moment])), 2)
            diff_test_AngVel += math.pow((float(to_show_guessed [1, moment])) - (float(to_show_wanted [1, moment])), 2)
        loss = loss_function(tensor_w, tensor_g)
        
        losses_test_Vel[idx] = diff_test_Vel / consts.LEN_OF_SEGMENTS
        losses_test_AngVel[idx] = diff_test_AngVel / consts.LEN_OF_SEGMENTS
        losses_per_test_segment[idx] = loss
        
        idx += 1
    
    if train:
        plt.figure(figure_const *10000000000 + 2000000000 + WIDTH_OF_LAYERS *1000000 + NUM_OF_LAYERS *1000 + EPOCH)    
        plt.xlabel('train segment')
    else:
        plt.figure(figure_const *10000000000 + 1000000000 + WIDTH_OF_LAYERS *1000000 + NUM_OF_LAYERS *1000 + EPOCH)    
        plt.xlabel('test segment')
    plt.title(str("net layers (width; num): " + str(WIDTH_OF_LAYERS) + "; " + str(NUM_OF_LAYERS) + "; epoch: " + str(EPOCH) + "  MSE per seg in test Vel"))
    plt.ylabel('MSE')
    plt.plot(segments_l, losses_test_Vel)
    plt.legend(loc=2)
    plt.axis([segments_l[0], segments_l[-1], 0, 1])
    plt.show()
    
    if train:
        plt.figure(figure_const *10000000000 + 2000000000 + WIDTH_OF_LAYERS *1000000 + NUM_OF_LAYERS *1000 + EPOCH +1)    
        plt.xlabel('train segment')
    else:
        plt.figure(figure_const *10000000000 + 1000000000 + WIDTH_OF_LAYERS *1000000 + NUM_OF_LAYERS *1000 + EPOCH +1)    
        plt.xlabel('test segment')
    plt.title(str("net layers (width; num): " + str(WIDTH_OF_LAYERS) + "; " + str(NUM_OF_LAYERS) + "; epoch: " + str(EPOCH) + "  MSE per seg in test AngVel"))
    plt.ylabel('MSE')
    plt.plot(segments_l, losses_test_AngVel)
    plt.legend(loc=2)
    plt.axis([segments_l[0], segments_l[-1], 0, 1])
    plt.show()
    
    if train:
        plt.figure(figure_const *10000000000 + 2000000000 + WIDTH_OF_LAYERS *1000000 + NUM_OF_LAYERS *1000 + EPOCH +2)    
        plt.xlabel('train segment')
    else:
        plt.figure(figure_const *10000000000 + 1000000000 + WIDTH_OF_LAYERS *1000000 + NUM_OF_LAYERS *1000 + EPOCH +2)    
        plt.xlabel('test segment')
    plt.title(str("net layers (width; num): " + str(WIDTH_OF_LAYERS) + "; " + str(NUM_OF_LAYERS) + "; epoch: " + str(EPOCH) + "  loss per seg"))
    plt.ylabel('loss')
    plt.plot(segments_l, losses_per_test_segment)
    plt.legend(loc=2)
    plt.axis([segments_l[0], segments_l[-1], 0, 1])
    plt.show()
    
    print("Avgerage loss on these segments:", sum(losses_per_test_segment) / len(losses_per_test_segment))
    if train:
        print("Loaded loss on all train segments:", matrix_loaded_losses[EPOCH, 1])
    else:
        print("Loaded loss on all test segments:", matrix_loaded_losses[EPOCH, 2])

def loss(figure_const, widthes_l, numbers_of_layers_l, same=False):
    col_idx = 0
    style_idx = 0
    for w in widthes_l:
        for n in numbers_of_layers_l:
            m = np.load(FILE_PRE + FILE_NAME_START + 'losses_and_val_losses_of_{}_{}.npy'.format(w, n))
            if same:
                plt.figure(figure_const *10 + 0)    
                plt.title(FILE_PRE)
                plt.xlabel('epoch')
                plt.ylabel('loss')
                print(colours[col_idx])
                plt.plot(m[ : , 1], colours[col_idx], label=str(str(w) + "; " + str(n) + " train"))
                col_idx += 1
                print(colours[col_idx])
                plt.plot(m[ : , 2], colours[col_idx], label=str(str(w) + "; " + str(n) + " test"))
                col_idx += 1
                plt.legend(loc=1)
                plt.axis([-5, consts.EPOCHS, 0, 1.5])
                plt.show()
            
            else:
                plt.figure(figure_const *10 + 1)    
                plt.title(FILE_PRE)
                plt.xlabel('epoch')
                plt.ylabel('loss')
                print(w, n, style_idx)
                print(colours[col_idx])
                print(styles[style_idx])
                plt.plot(m[ : , 2], colours[col_idx], label=str(str(w) + "; " + str(n) + " test"), linestyle=styles[style_idx])
                plt.legend(loc=1)
                plt.axis([-5, consts.EPOCHS, 0, 1.5])
                plt.show()
                
                plt.figure(figure_const *10 + 2)    
                plt.title(FILE_PRE)
                plt.xlabel('epoch')
                plt.ylabel('loss')
                print(w, n, style_idx)
                print(colours[col_idx])
                print(styles[style_idx])
                plt.plot(m[ : , 1], colours[col_idx], label=str(str(w) + "; " + str(n) + " train"), linestyle=styles[style_idx])
                plt.legend(loc=1)
                plt.axis([-5, consts.EPOCHS, 0, 1.5])
                plt.show()
                
                col_idx += 1
        style_idx += 1
        
def rmse(figure_const, widthes_l, numbers_of_layers_l):
    col_idx = 0
    for w in widthes_l:
        for n in numbers_of_layers_l:
            m = np.load(FILE_PRE + FILE_NAME_START + 'losses_and_val_losses_of_{}_{}.npy'.format(w, n))
            plt.figure(figure_const *10 + 0)    
            plt.title(str("net layers (width; num): " + str(w) + "; " + str(n)))
            plt.xlabel('epoch')
            plt.ylabel('loss')
            print(colours[col_idx])
            plt.plot(np.sqrt(m[ : , 1]), colours[col_idx], label=str(str(w) + "; " + str(n) + " train"))
            col_idx += 1
            plt.plot(np.sqrt(m[ : , 2]), colours[col_idx], label=str(str(w) + "; " + str(n) + " test"))
            col_idx += 1
            plt.legend(loc=1)
            plt.axis([-5, consts.EPOCHS, 0, 3.5])
            plt.show()
                
def loss_table(figure_const, widthes_l, numbers_of_layers_l):
    matrix_w_n_losses = np.zeros([len(widthes_l), len(numbers_of_layers_l), 2], dtype=float)
    i = 0
    for w in widthes_l:
        j = 0
        for n in numbers_of_layers_l:
            m = np.load(FILE_PRE + FILE_NAME_START + 'losses_and_val_losses_of_{}_{}.npy'.format(w, n))
            length = 10
            matrix_w_n_losses[i, j, 0] = sum(m[consts.EPOCHS -length : consts.EPOCHS, 1]) / length
            matrix_w_n_losses[i, j, 1] = sum(m[consts.EPOCHS -length : consts.EPOCHS, 2]) / length
            j += 1
    
        plt.figure(figure_const *10 + 3)    
        plt.title("train loss table" + FILE_PRE)
        plt.xlabel('number of layers')
        plt.ylabel('loss')
        plt.plot(numbers_of_layers_l, matrix_w_n_losses[ i, : , 0], label="train " + str(w))
        plt.legend(loc=1)
        plt.axis([numbers_of_layers_l[0], numbers_of_layers_l[-1], 0, 0.5])
        plt.show()
        
        plt.figure(figure_const *10 + 4)
        plt.xlabel('number of layers')
        plt.ylabel('loss')
        plt.title("test loss table of " + FILE_PRE)
        plt.plot(numbers_of_layers_l, matrix_w_n_losses[ i, : , 1], label="test " + str(w))
        plt.legend(loc=1)
        plt.axis([numbers_of_layers_l[0], numbers_of_layers_l[-1], 0, 0.5])
        plt.show()
        
        i += 1
        
    
# ehhez nem kell nyúlni
##############################################################################


##############################################################################
# itt állítjuk be, amire kíváncsiak vagyunk
    # választott szegmensek
test_segments = [1, 2] # range(27)           # matrix_test_glob-ból
train_segments = [1, 2] # range(851)          # matrix_train_glob-ból
widthes = [24] # np.array([8, 16, 24])
numbers_of_layers = [2] # np.array([2, 4, 6, 8, 10])
    # szegmensek pillanatonként
segments_test(2, test_segments, train_segments)
    # loss check
loss_check(1, test_segments, False)
loss_check(2, train_segments, True)
loss(7, widthes, numbers_of_layers, False)
#rmse(111, widthes, numbers_of_layers)
#loss_table(6, widthes, numbers_of_layers)
