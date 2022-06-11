import scipy.io
import matplotlib.pyplot as plt
#import math

# adat elokeszites
# DATA_STRIDE = 10
# mat = scipy.io.loadmat("meas01.mat")
# MEAS_LEN = 10119
# mat = scipy.io.loadmat("meas02.mat")
# MEAS_LEN = 10306
mat_cc = scipy.io.loadmat("meas_cornercut03_speedcontrol5__X_Y_Ori_Vel_AngVel_WFs_SW_DevAng00051015_DevDist00051015_TO_PATH_RouteXYZ_Path_XYZ.mat")
#mat_nc = scipy.io.loadmat("meas_cornercut0_speed2__X_Y_Ori_Vel_AngVel_WFs_SW_DevAng00051015_DevDist00051015_TO_PATH_RouteXYZ_Path_XYZ.mat")


# Kimenetek
SW_cc = mat_cc["SteAng"]

plt.figure(10)
plt.title("SW cc")
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
DevAng00_cc = mat_cc["DevAng00"]
DevDist05_cc = mat_cc["DevDist05"]
DevAng05_cc = mat_cc["DevAng05"]
DevDist10_cc = mat_cc["DevDist10"]
DevAng10_cc = mat_cc["DevAng10"]
DevDist15_cc = mat_cc["DevDist15"]
DevAng15_cc = mat_cc["DevAng15"]


# plt.figure(1)
# plt.title("DevDist00 cc")
# plt.plot(DevDist00_cc, 'b')
# plt.show()

# plt.figure(2)
# plt.title("DevDist05 cc")
# plt.plot(DevDist05_cc, 'b')
# plt.show()

# plt.figure(3)
# plt.title("DevDist10 cc")
# plt.plot(DevDist10_cc, 'b')
# plt.show()

# plt.figure(4)
# plt.title("DevDist15 cc")
# plt.plot(DevDist15_cc, 'b')
# plt.show()

plt.figure(5)
plt.title("DevDist00 05 10 15 cc")
plt.plot(DevDist00_cc, 'blue', label='00')
plt.plot(DevDist05_cc, 'red', label='05')
plt.plot(DevDist10_cc, 'orange', label='10')
plt.plot(DevDist15_cc, 'green', label='15')
plt.legend(loc=1)
plt.show()

# plt.figure(6)
# plt.title("DevAng00 cc")
# plt.plot(DevAng00_cc, 'b')
# plt.show()

# plt.figure(7)
# plt.title("DevAng05 cc")
# plt.plot(DevAng05_cc, 'b')
# plt.show()

# plt.figure(8)
# plt.title("DevAng10 cc")
# plt.plot(DevAng10_cc, 'b')
# plt.show()

# plt.figure(9)
# plt.title("DevAng15 cc")
# plt.plot(DevAng15_cc, 'b')
# plt.show()

# plt.figure(99)
# plt.title("DevAng00 05 10 15 cc")
# plt.plot(DevAng00_cc, 'blue', label='00')
# plt.plot(DevAng05_cc, 'red', label='05')
# plt.plot(DevAng10_cc, 'orange', label='10')
# plt.plot(DevAng15_cc, 'green', label='15')
# plt.legend(loc=1)
# plt.show()

plt.figure(100)
plt.title("DevAng00 05 10 15 + SW cc")
plt.plot(DevAng00_cc, 'blue', label='00')
plt.plot(DevAng05_cc, 'red', label='05')
plt.plot(DevAng10_cc, 'orange', label='10')
plt.plot(DevAng15_cc, 'green', label='15')
plt.plot(SW_cc, 'black', label='SW')
plt.legend(loc=1)
plt.show()


# # Palya
# plt.figure(12)
# plt.title("pos cc")
# plt.plot(Pos_X_cc, Pos_Y_cc, 'b')
# plt.show()

# plt.figure(13)
# plt.title("route cc")
# plt.plot(RouteX_cc, RouteY_cc, 'b')
# plt.show()

# plt.figure(14)
# plt.title("path cc")
# plt.plot(PathX_cc, PathY_cc, 'b')
# plt.show()

# plt.figure(15)
# plt.title("all cc")
# plt.plot(Pos_X_cc, Pos_Y_cc, 'red', label='pos')
# plt.plot(RouteX_cc, RouteY_cc, 'green', label='route')
# plt.plot(PathX_cc, PathY_cc, 'blue', label='path')
# plt.legend(loc=1)
# plt.show()

# plt.figure(16)
# plt.title("pos vs path cc")
# plt.plot(Pos_X_cc, 'red', label='posX')
# plt.plot(Pos_Y_cc, 'orange', label='posY')
# plt.plot(PathX_cc, 'blue', label='pathX')
# plt.plot(PathY_cc, 'lightblue', label='pathY')
# plt.legend(loc=1)
# plt.show()

#############################################################
# innentol nc

# # Kimenetek
# SW_nc = mat_nc["SteAng"]

# plt.figure(20)
# plt.title("SW nc")
# plt.plot(SW_nc, 'b')
# plt.show()


# # Bemenetek
# Pos_X_nc = mat_nc["Pos_X"]
# Pos_Y_nc = mat_nc["Pos_Y"]
# RouteX_nc = mat_nc["RouteX"]
# RouteY_nc = mat_nc["RouteY"]
# PathX_nc = mat_nc["PathX"]
# PathY_nc = mat_nc["PathY"]
# DevDist00_nc = mat_nc["DevDist00"]


# plt.figure(21)
# plt.title("DevDist00 nc")
# plt.plot(DevDist00_nc, 'b')
# plt.show()

# # Palya
# plt.figure(22)
# plt.title("pos nc")
# plt.plot(Pos_X_nc, Pos_Y_nc, 'b')
# plt.show()

# plt.figure(23)
# plt.title("route nc")
# plt.plot(RouteX_nc, RouteY_nc, 'b')
# plt.show()

# plt.figure(24)
# plt.title("path nc")
# plt.plot(PathX_nc, PathY_nc, 'b')
# plt.show()

# plt.figure(25)
# plt.title("all nc")
# plt.plot(Pos_X_nc, Pos_Y_nc, 'red', label='pos')
# plt.plot(RouteX_nc, RouteY_nc, 'green', label='route')
# plt.plot(PathX_nc, PathY_nc, 'blue', label='path')
# plt.legend(loc=1)
# plt.show()

# plt.figure(26)
# plt.title("pos vs path nc")
# plt.plot(Pos_X_nc, 'red', label='posX')
# plt.plot(Pos_Y_nc, 'orange', label='posY')
# plt.plot(PathX_nc, 'blue', label='pathX')
# plt.plot(PathY_nc, 'lightblue', label='pathY')
# plt.legend(loc=1)
# plt.show()


# diff = [None] * len(Pos_X_nc)
# for m in range(len(Pos_X_nc)):
#     vx = Pos_X_nc[m] - PathX_nc[m]
#     vy = Pos_Y_nc[m] - PathY_nc[m]
#     vlength = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
#     diff[m] = vlength + DevDist00_nc[m]
    
# plt.figure(30)
# plt.title("diff path nc")
# plt.plot(diff)
# plt.show()

# diff = [None] * len(Pos_X_nc)
# for m in range(len(Pos_X_nc)):
#     vx = Pos_X_nc[m] - RouteX_nc[m]
#     vy = Pos_Y_nc[m] - RouteY_nc[m]
#     vlength = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
#     diff[m] = vlength + DevDist00_nc[m]
    
# plt.figure(31)
# plt.title("diff route nc")
# plt.plot(diff)
# plt.show()

# diff = [None] * len(Pos_X_nc)
# for m in range(len(Pos_X_nc)):
#     vx = Pos_X_nc[m] - PathX_nc[m]
#     vy = Pos_Y_nc[m] - PathY_nc[m]
#     vlength = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
#     diff[m] = vlength + DevDist00_nc[m]
    
# plt.figure(40)
# plt.title("diff path nc")
# plt.plot(diff)
# plt.show()

# diff = [None] * len(Pos_X_nc)
# for m in range(len(Pos_X_nc)):
#     vx = Pos_X_nc[m] - RouteX_nc[m]
#     vy = Pos_Y_nc[m] - RouteY_nc[m]
#     vlength = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
#     diff[m] = vlength + DevDist00_nc[m]
    
# plt.figure(41)
# plt.title("diff route nc")
# plt.plot(diff)
# plt.show()
