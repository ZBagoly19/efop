import scipy.io
import numpy as np


filename = "data_to_zoli.mat"
mat = scipy.io.loadmat(filename)
train_input = mat["data_train_input_1"]
train_label = mat["data_train_output"]
test_input = mat["data_test_input_1"]
test_label = mat["data_test_output"]

train_data = [None] * len(train_input[0][0])
testing_data = [None] * len(test_input[0][0])

for data_point_idx1 in range(len(train_input[0][0])):
    train_data[data_point_idx1] = [
        train_input[:, :, data_point_idx1].reshape(100, 1),
        train_label[data_point_idx1],
        data_point_idx1
    ]
for data_point_idx2 in range(len(test_input[0][0])):
    testing_data[data_point_idx2] = [
        test_input[:, :, data_point_idx2].reshape(100, 1),
        test_label[data_point_idx2],
        data_point_idx2
    ]

np.random.shuffle(testing_data)
np.save("Mate_data_test_part__" + ".npy", testing_data)
np.random.shuffle(train_data)
np.save("Mate_data_train_part__" + ".npy", train_data)
