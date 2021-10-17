import scipy.io
import matplotlib.pyplot as plt
import torch

device = torch.device("cpu")

# GPU
def on_gpu():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

device = on_gpu()
print(device)

# adat elokeszites
REBUILD_DATA = False
my_training_data = []
my_testing_data = []
LEN_OF_SEGMENTS = 400
LEN_OF_INPUT = 4 * LEN_OF_SEGMENTS
LEN_OF_OUTPUT = 5 * LEN_OF_SEGMENTS
DATA_STRIDE = 10
mat = scipy.io.loadmat("meas01.mat")

# Kimenetek
Fy_FL = mat["WheForceX_FL"]
Fy_FR = mat["WheForceX_FR"]
Fy_RL = mat["WheForceX_RL"]
Fy_RR = mat["WheForceX_RR"]
WheelAngle = mat["WheAng"]

times = []
for i in range(10119):
    times.append(i)
    
fig = plt.figure()

ax1 = plt.subplot2grid((5, 1), (0, 0))
ax1.plot(times, Fy_FL, 'b', label="Fy_FL")
ax1.legend(loc=2)

ax1 = plt.subplot2grid((5, 1), (1, 0))
ax1.plot(times, Fy_FR, 'b', label="Fy_FR")
ax1.legend(loc=2)

ax1 = plt.subplot2grid((5, 1), (2, 0))
ax1.plot(times, Fy_RL, 'b', label="Fy_RL")
ax1.legend(loc=2)

ax1 = plt.subplot2grid((5, 1), (3, 0))
ax1.plot(times, Fy_RR, 'b', label="Fy_RR")
ax1.legend(loc=2)

ax1 = plt.subplot2grid((5, 1), (4, 0))
ax1.plot(times, WheelAngle, 'b', label="WheelAngle")
ax1.legend(loc=2)


# Bemenetek
Orientation = mat["Ori_Z"]
Pos_X = mat["Pos_X"]
Pos_Y = mat["Pos_Y"]
Velocity = mat["Vel"]

fig = plt.figure()

ax2 = plt.subplot2grid((4, 1), (0, 0))
ax2.plot(times, Orientation, 'b', label="Ori")
ax2.legend(loc=2)

ax2 = plt.subplot2grid((4, 1), (1, 0))
ax2.plot(times, Pos_X, 'b', label="Pos_X")
ax2.legend(loc=2)

ax2 = plt.subplot2grid((4, 1), (2, 0))
ax2.plot(times, Pos_Y, 'b', label="Pos_Y")
ax2.legend(loc=2)

ax2 = plt.subplot2grid((4, 1), (3, 0))
ax2.plot(times, Velocity, 'b', label="Velo")
ax2.legend(loc=2)


# Palya??
fig = plt.figure()

ax3 = plt.subplot2grid((1, 1), (0, 0))
ax3.plot(Pos_X, Pos_Y, 'b', label="track")
ax3.legend(loc=2)