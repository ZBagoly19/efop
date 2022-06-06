"""
@author: Bagoly Zoltán
"""
import numpy as np
import SZTAKI_constants as consts

##############################################################################
# itt állítjuk be, mi érdekel
file_name_start = 'XYO_VAV__'
MODEL_NUMBER = 16
NETS = np.array([2, 4, 6, 8, 10, 12, 14])      # layerek szama

##############################################################################
# ehhez nem kell nyúlni
loaded_losses = np.zeros([len(NETS), consts.EPOCHS, 3], dtype=float)
matrix_test_glob = np.zeros([len(NETS), consts.EPOCHS, consts.NUM_OF_TESTS, 2, consts.NUM_OF_OUTPUT_DATA_TYPES, consts.LEN_OF_SEGMENTS], dtype=float)
i = 0
for n in NETS:
    loaded_losses[i] = np.load(file_name_start + 'losses_and_val_losses_of_{}_{}.npy'.format(MODEL_NUMBER, n))
    matrix_test_glob[i] = np.load(file_name_start + 'matrices_of_{}_{}.npy'.format(MODEL_NUMBER, n))
    i += 1
