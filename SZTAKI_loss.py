"""
@author: Bagoly Zoltán
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import SZTAKI_constants as consts
import SZTAKI_load_test_matrices as loaded
style.use("ggplot")

##############################################################################
# itt állítjuk be, mi érdekel
NETS = np.array([2, 4, 6, 8, 10, 12, 14])      # numbers of layers
both = False

##############################################################################
# ehhez nem kell nyúlni

for i in range(len(NETS)):
    plt.figure(loaded.MODEL_NUMBER*10000)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(loaded.loaded_losses[i, :, 1], label=str(str(NETS[i]) + " train"))
    # plt.plot(loaded.loaded_losses[i, :, 2], label=str(str(NETS[i]) + " val"))
    plt.legend(loc=1)
    plt.axis([-5, consts.EPOCHS, 0, 3.5])
plt.show()

for i in range(len(NETS)):
    plt.figure(loaded.MODEL_NUMBER*10000+1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.plot(loaded.loaded_losses[i, :, 1], label=str(str(NETS[i]) + " train"))
    plt.plot(loaded.loaded_losses[i, :, 2], label=str(str(NETS[i]) + " val"))
    plt.legend(loc=1)
    plt.axis([-5, consts.EPOCHS, 0, 3.5])
plt.show()

if both:
    for i in range(len(NETS)):
        plt.figure(loaded.MODEL_NUMBER*10000+2)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(loaded.loaded_losses[i, :, 1], label=str(str(NETS[i]) + " train"))
        plt.plot(loaded.loaded_losses[i, :, 2], label=str(str(NETS[i]) + " val"))
        plt.legend(loc=1)
        plt.axis([-5, consts.EPOCHS, 0, 3.5])
    plt.show()