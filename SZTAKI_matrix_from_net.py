"""
@author: Bagoly Zoltán
"""
import os
import numpy as np
import tqdm as my_tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import SZTAKI_constants as consts
DEVICE = torch.device("cpu")

##############################################################################
# itt állítjuk be, melyik hálóhoz generáljunk mátrixokat
# NET_TYPE =   "fully connected"   # "convolutional" or "fully connected"
file_name_start = 'net_XYO_VAV__'
MODEL_NUMBER = 16
NETS = np.array([2, 4, 6, 8, 10, 12, 14])      # layerek szama
train_matrices_as_well = True   # ha True, a tanító adatokra adott válaszokat is megkapjuk

##############################################################################
# ehhez nem kell nyúlni

# GPU
def on_gpu():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

DEVICE = on_gpu()

def load_net(path_param):
    if os.path.isfile(path_param):
        print("helyes filenev:", path_param)
        
        net = torch.load(os.path.join(path_param))
        print(net)
        net.eval()

loss_function = nn.MSELoss()

LEN_OF_INPUT = consts.NUM_OF_INPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS
LEN_OF_OUTPUT = consts.NUM_OF_OUTPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS

SHUF_TEST_DATA = np.load("my_testing_data_XYO_VAV_better_skale.npy", allow_pickle=True)
SHUF_TRAIN_DATA = np.load("my_training_data_XYO_VAV_better_skale.npy", allow_pickle=True)
TRAIN_X = torch.Tensor([i[0] for i in SHUF_TRAIN_DATA]).view(-1, consts.NUM_OF_INPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS)
TEST_X = torch.Tensor([i[0] for i in SHUF_TEST_DATA]).view(-1, consts.NUM_OF_INPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS)
TRAIN_Y = torch.Tensor([i[1] for i in SHUF_TRAIN_DATA]).view(-1, consts.NUM_OF_OUTPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS)
TEST_Y = torch.Tensor([i[1] for i in SHUF_TEST_DATA]).view(-1, consts.NUM_OF_OUTPUT_DATA_TYPES * consts.LEN_OF_SEGMENTS)

matrix_test_glob = np.zeros([len(NETS), consts.EPOCHS, len(TEST_X), 2, consts.NUM_OF_OUTPUT_DATA_TYPES, consts.LEN_OF_SEGMENTS], dtype=float)
matrix_train_glob = np.zeros([len(NETS), consts.EPOCHS, len(TRAIN_X), 2, consts.NUM_OF_OUTPUT_DATA_TYPES, consts.LEN_OF_SEGMENTS], dtype=float)
net_idx = 0
for n in NETS:
    # Net fully connected
    class Net_fc(nn.Module):
        def __init__(self):
            super().__init__()
            
            x = torch.randn(LEN_OF_INPUT).view(-1, 1, LEN_OF_INPUT)
            self._to_linear = None
            self.linearize(x)
            
            self.fc1 = nn.Linear(self._to_linear, MODEL_NUMBER)
            nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
            self.fc2 = nn.Linear(MODEL_NUMBER, MODEL_NUMBER)
            nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
            if 2 < n:
                self.fc3 = nn.Linear(MODEL_NUMBER, MODEL_NUMBER)
                nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
                self.fc4 = nn.Linear(MODEL_NUMBER, MODEL_NUMBER)
                nn.init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
            if 4 < n:
                self.fc5 = nn.Linear(MODEL_NUMBER, MODEL_NUMBER)
                nn.init.kaiming_uniform_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
                self.fc6 = nn.Linear(MODEL_NUMBER, MODEL_NUMBER)
                nn.init.kaiming_uniform_(self.fc6.weight, mode='fan_in', nonlinearity='relu')
            if 6 < n:
                self.fc7 = nn.Linear(MODEL_NUMBER, MODEL_NUMBER)
                nn.init.kaiming_uniform_(self.fc7.weight, mode='fan_in', nonlinearity='relu')
                self.fc8 = nn.Linear(MODEL_NUMBER, MODEL_NUMBER)
                nn.init.kaiming_uniform_(self.fc8.weight, mode='fan_in', nonlinearity='relu')
            if 8 < n:
                self.fc9 = nn.Linear(MODEL_NUMBER, MODEL_NUMBER)
                nn.init.kaiming_uniform_(self.fc9.weight, mode='fan_in', nonlinearity='relu')
                self.fc10 = nn.Linear(MODEL_NUMBER, MODEL_NUMBER)
                nn.init.kaiming_uniform_(self.fc10.weight, mode='fan_in', nonlinearity='relu')
            if 10 < n:
                self.fc11 = nn.Linear(MODEL_NUMBER, MODEL_NUMBER)
                nn.init.kaiming_uniform_(self.fc11.weight, mode='fan_in', nonlinearity='relu')
                self.fc12 = nn.Linear(MODEL_NUMBER, MODEL_NUMBER)
                nn.init.kaiming_uniform_(self.fc12.weight, mode='fan_in', nonlinearity='relu')
            if 12 < n:
                self.fc13 = nn.Linear(MODEL_NUMBER, MODEL_NUMBER)
                nn.init.kaiming_uniform_(self.fc13.weight, mode='fan_in', nonlinearity='relu')
                self.fc14 = nn.Linear(MODEL_NUMBER, MODEL_NUMBER)
                nn.init.kaiming_uniform_(self.fc14.weight, mode='fan_in', nonlinearity='relu')
            self.fc_last = nn.Linear(MODEL_NUMBER, LEN_OF_OUTPUT)
            nn.init.kaiming_uniform_(self.fc_last.weight, mode='fan_in', nonlinearity='relu')
        def linearize(self, x):
            if self._to_linear == None:
                self._to_linear = x[0].shape[0] * x[0].shape[1]
            return x
            
        def forward(self, x):        
            x = x.view(-1, self._to_linear)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            if 2 < n:
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
            if 4 < n:
                x = F.relu(self.fc5(x))
                x = F.relu(self.fc6(x))
            if 6 < n:
                x = F.relu(self.fc7(x))
                x = F.relu(self.fc8(x))
            if 8 < n:
                x = F.relu(self.fc9(x))
                x = F.relu(self.fc10(x))
            if 10 < n:
                x = F.relu(self.fc11(x))
                x = F.relu(self.fc12(x))
            if 12 < n:
                x = F.relu(self.fc13(x))
                x = F.relu(self.fc14(x))
            x = self.fc_last(x)        
            return x
    for epoch in range(consts.EPOCH_SAVE_CONST - 1, consts.EPOCHS, consts.EPOCH_SAVE_CONST):
        print("próba:", 'net_XYO_VAV__{}_{}_{}.pth'.format(MODEL_NUMBER, n, epoch))
        if os.path.isfile(os.path.join('net_XYO_VAV__{}_{}_{}.pth'.format(MODEL_NUMBER, n, epoch))):
            print("betöltött háló", os.path.join('net_XYO_VAV__{}_{}_{}.pth'.format(MODEL_NUMBER, n, epoch)))
            
            net = torch.load(os.path.join('net_XYO_VAV__{}_{}_{}.pth'.format(MODEL_NUMBER, n, epoch)))
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
                
            def one_segment_test(start, print_loss=False):
                my_X3, my_y3 = TEST_X[start : start + 1], TEST_Y[start : start + 1]
                to_show_wanted = my_y3.to(DEVICE)
                to_show_guessed = net(my_X3.view(-1, 1, LEN_OF_INPUT).to(DEVICE))
                loss = loss_function(to_show_guessed, to_show_wanted)
                if print_loss:
                    print(loss)
                return to_show_wanted, to_show_guessed
            def one_segment_test_on_train(start):
                my_X3, my_y3 = TRAIN_X[start : start + 1], TRAIN_Y[start : start + 1]
                to_show_wanted = my_y3.to(DEVICE)
                to_show_guessed = net(my_X3.view(-1, 1, LEN_OF_INPUT).to(DEVICE))
                return to_show_wanted, to_show_guessed
            
            def create_matrices(train_as_well=False):
                matrix_test = np.zeros([len(TEST_X), LEN_OF_OUTPUT * 2], dtype=float)
                for seg in my_tqdm.tqdm(range(0, len(TEST_X), 1)):
                    wanted, guessed = one_segment_test(seg)
                    a = wanted.cpu()
                    b = guessed.cpu()
                    c = a.detach().numpy()
                    d = b.detach().numpy()
                    matrix_row = np.column_stack([c, d])
                    matrix_test[seg] = matrix_row
                matrix_test_l_formed = np.zeros([len(TEST_X), 2, consts.NUM_OF_OUTPUT_DATA_TYPES, consts.LEN_OF_SEGMENTS], dtype=float)
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
            
                matrix_train = np.zeros([len(TRAIN_X), LEN_OF_OUTPUT * 2], dtype=float)
                matrix_train_l_formed = np.zeros([len(TRAIN_X), 2, consts.NUM_OF_OUTPUT_DATA_TYPES, consts.LEN_OF_SEGMENTS], dtype=float)
                if train_as_well:
                    for seg in my_tqdm.tqdm(range(0, len(TRAIN_X), 1)):
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
            matrix_test_g, matrix_train_g = create_matrices(train_matrices_as_well)
            matrix_test_glob[net_idx, epoch] = matrix_test_g
            matrix_train_glob[net_idx, epoch] = matrix_train_g
    net_idx += 1
