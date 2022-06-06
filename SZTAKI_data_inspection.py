import scipy.io
import matplotlib.pyplot as plt
import math

# adat elokeszites
# DATA_STRIDE = 10
# mat = scipy.io.loadmat("meas01.mat")
# MEAS_LEN = 10119
# mat = scipy.io.loadmat("meas02.mat")
# MEAS_LEN = 10306
mat_cc = scipy.io.loadmat("meas_cornercut03__X_Y_Ori_Vel_AngVel_WFs_SW_DevAng00051015_DevDist00051015_RouteXYZ_Path_XYZ.mat")
mat_nc = scipy.io.loadmat("meas_X_Y_Ori_Vel_AngVel_WFs_SW_DevAng00051015_DevDist00051015_RouteXYZ_Path_XYZ.mat")


# Kimenetek
SW_cc = mat_cc["SteAng"]

plt.figure(10)
plt.title("SW")
plt.plot(SW_cc, 'b')
plt.show()


# Bemenetek
Pos_X_cc = mat_cc["Pos_X"]
Pos_Y_cc = mat_cc["Pos_Y"]
RouteX_cc = mat_cc["RouteX"]
RouteY_cc = mat_cc["RouteY"]
PathX_cc = mat_cc["PathX"]
PathY_cc = mat_cc["PathY"]

DevDist00_cc = mat_cc["DevDist00"]


plt.figure(11)
plt.title("DevDist00 cc")
plt.plot(DevDist00_cc, 'b')
plt.show()

# Palya
plt.figure(12)
plt.title("pos cc")
plt.plot(Pos_X_cc, Pos_Y_cc, 'b')
plt.show()

plt.figure(13)
plt.title("route cc")
plt.plot(RouteX_cc, RouteY_cc, 'b')
plt.show()

plt.figure(14)
plt.title("path cc")
plt.plot(PathX_cc, PathY_cc, 'b')
plt.show()

plt.figure(15)
plt.title("all cc")
plt.plot(Pos_X_cc, Pos_Y_cc, 'red', label='pos')
plt.plot(RouteX_cc, RouteY_cc, 'green', label='route')
plt.plot(PathX_cc, PathY_cc, 'blue', label='path')
plt.legend(loc=1)
plt.show()

plt.figure(16)
plt.title("pos vs path cc")
plt.plot(Pos_X_cc, 'red', label='posX')
plt.plot(Pos_Y_cc, 'orange', label='posY')
plt.plot(PathX_cc, 'blue', label='pathX')
plt.plot(PathY_cc, 'lightblue', label='pathY')
plt.legend(loc=1)
plt.show()

#############################################################

# Kimenetek
SW_nc = mat_nc["SteAng"]

plt.figure(20)
plt.title("SW")
plt.plot(SW_nc, 'b')
plt.show()


# Bemenetek
Pos_X_nc = mat_nc["Pos_X"]
Pos_Y_nc = mat_nc["Pos_Y"]
RouteX_nc = mat_nc["RouteX"]
RouteY_nc = mat_nc["RouteY"]
PathX_nc = mat_nc["PathX"]
PathY_nc = mat_nc["PathY"]
DevDist00_nc = mat_nc["DevDist00"]


plt.figure(21)
plt.title("DevDist00 nc")
plt.plot(DevDist00_nc, 'b')
plt.show()

# Palya
plt.figure(22)
plt.title("pos nc")
plt.plot(Pos_X_nc, Pos_Y_nc, 'b')
plt.show()

plt.figure(23)
plt.title("route nc")
plt.plot(RouteX_nc, RouteY_nc, 'b')
plt.show()

plt.figure(24)
plt.title("path nc")
plt.plot(PathX_nc, PathY_nc, 'b')
plt.show()

plt.figure(25)
plt.title("all nc")
plt.plot(Pos_X_nc, Pos_Y_nc, 'red', label='pos')
plt.plot(RouteX_nc, RouteY_nc, 'green', label='route')
plt.plot(PathX_nc, PathY_nc, 'blue', label='path')
plt.legend(loc=1)
plt.show()

plt.figure(26)
plt.title("pos vs path nc")
plt.plot(Pos_X_nc, 'red', label='posX')
plt.plot(Pos_Y_nc, 'orange', label='posY')
plt.plot(PathX_nc, 'blue', label='pathX')
plt.plot(PathY_nc, 'lightblue', label='pathY')
plt.legend(loc=1)
plt.show()


diff = [None] * len(Pos_X_nc)
for m in range(len(Pos_X_nc)):
    vx = Pos_X_nc[m] - PathX_nc[m]
    vy = Pos_Y_nc[m] - PathY_nc[m]
    vlength = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
    diff[m] = vlength + DevDist00_nc[m]
    
plt.figure(30)
plt.title("diff path nc")
plt.plot(diff)
plt.show()

diff = [None] * len(Pos_X_nc)
for m in range(len(Pos_X_nc)):
    vx = Pos_X_nc[m] - RouteX_nc[m]
    vy = Pos_Y_nc[m] - RouteY_nc[m]
    vlength = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
    diff[m] = vlength + DevDist00_nc[m]
    
plt.figure(31)
plt.title("diff route nc")
plt.plot(diff)
plt.show()

diff = [None] * len(Pos_X_nc)
for m in range(len(Pos_X_nc)):
    vx = Pos_X_nc[m] - PathX_nc[m]
    vy = Pos_Y_nc[m] - PathY_nc[m]
    vlength = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
    diff[m] = vlength + DevDist00_nc[m]
    
plt.figure(40)
plt.title("diff path cc")
plt.plot(diff)
plt.show()

diff = [None] * len(Pos_X_nc)
for m in range(len(Pos_X_nc)):
    vx = Pos_X_nc[m] - RouteX_nc[m]
    vy = Pos_Y_nc[m] - RouteY_nc[m]
    vlength = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
    diff[m] = vlength + DevDist00_nc[m]
    
plt.figure(41)
plt.title("diff route cc")
plt.plot(diff)
plt.show()
