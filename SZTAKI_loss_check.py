"""
@author: Bagoly Zoltán
"""
import math
import numpy as np
import tqdm as my_tqdm
import matplotlib.pyplot as plt
from matplotlib import style
import torch
import SZTAKI_constants as consts
import SZTAKI_load_test_matrices as loaded
style.use("ggplot")
loss_function = torch.nn.MSELoss()

##############################################################################
# itt állítjuk be, mi érdekel
# ezekkel adjuk meg, mely fájlokat töltjük be
l = 'XYO_VAV__losses_and_val_losses_of_'
m = 'XYO_VAV__matrices_of_'
NETS = np.array([2, 4, 6, 8, 10, 12, 14])      # numbers of layers
# ezzel pedig, hogy miről rajzoljon
net, epoch = 0, 299                            # net: a NETS-ben az indexe a kívánt hálónak

##############################################################################
# ehhez nem kell nyúlni
if len(NETS) < net:
    print("Túl magas a net érték!")
if consts.EPOCHS < epoch:
    print("Túl magas a epoch érték!")

losses_val_Vel = np.zeros([len(loaded.matrix_test_glob[0][0])], dtype=float)
losses_val_AngVel = np.zeros([len(loaded.matrix_test_glob[0][0])], dtype=float)
losses_per_segment = np.zeros([len(loaded.matrix_test_glob[0][0])], dtype=float)

for seg in my_tqdm.tqdm(range(len(loaded.matrix_test_glob[0][0]))):
    #print(seg)
    to_show_wanted = loaded.matrix_test_glob[net, epoch, seg, 0]
    to_show_guessed = loaded.matrix_test_glob[net, epoch, seg, 1]
    tensor_w = torch.from_numpy(to_show_wanted)
    tensor_g = torch.from_numpy(to_show_guessed)                

    diff_val_Vel = 0
    diff_val_AngVel = 0

    for moment in range(consts.LEN_OF_SEGMENTS):
        diff_val_Vel += math.pow((float(to_show_guessed [0, moment])) - (float(to_show_wanted [0, moment])), 2)
        diff_val_AngVel += math.pow((float(to_show_guessed [1, moment])) - (float(to_show_wanted [1, moment])), 2)
    loss = loss_function(tensor_w, tensor_g)
    
    losses_val_Vel[seg] = diff_val_Vel / consts.LEN_OF_SEGMENTS
    losses_val_AngVel[seg] = diff_val_AngVel / consts.LEN_OF_SEGMENTS
    losses_per_segment[seg] = loss

plt.figure(NETS[net] *1000 + epoch)
plt.title(str("net layers: "+ str(NETS[net]) + "; epoch: " + str(epoch) + "  MSE per seg in test Vel"))
plt.xlabel('test segment')
plt.ylabel('MSE')
plt.plot(losses_val_Vel)
plt.legend(loc=2)
plt.axis([0, len(loaded.matrix_test_glob[0][0]), 0, 2])
plt.show()

plt.figure(NETS[net] *1000 + epoch +1)    
plt.title(str("net layers: "+ str(NETS[net]) + "; epoch: " + str(epoch) + "  MSE per seg in test AngVel"))
plt.xlabel('test segment')
plt.ylabel('MSE')
plt.plot(losses_val_AngVel)
plt.legend(loc=2)
plt.axis([0, len(loaded.matrix_test_glob[0][0]), 0, 2])
plt.show()

plt.figure(NETS[net] *1000 + epoch +2)    
plt.title(str("net layers: "+ str(NETS[net]) + "; epoch: " + str(epoch) + "  loss per seg"))
plt.xlabel('test segment')
plt.ylabel('loss')
plt.plot(losses_per_segment)
plt.legend(loc=2)
plt.axis([0, len(loaded.matrix_test_glob[0][0]), 0, 2])
plt.show()

print("Avgerage loss:", sum(losses_per_segment) / len(losses_per_segment))
print("Loaded   loss:", loaded.loaded_losses[net, epoch, 2])