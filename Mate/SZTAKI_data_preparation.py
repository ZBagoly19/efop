import scipy.io
import numpy as np


filename = "data_to_zoli.mat"
mat = scipy.io.loadmat(filename)
input_data = mat["data_train_input_1"]
label_data = mat["data_train_output"]
test_input = mat["data_test_input_1"]
test_label = mat["data_test_output"]

train_data = [None] * len(train_input[0][0])

for data_point_idx in range(len(input_data[0][0])):
    train_data[data_point_idx] = [
        input_data[:, :, data_point_idx].reshape(100, 1),
        label_data[data_point_idx],
        data_point_idx
    ]


np.random.shuffle(train_data)
np.save("Mate_data_train_part__" + ".npy", train_data)
