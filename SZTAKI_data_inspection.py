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

PathX1_nc = mat_nc["PathX1"]
PathY1_nc = mat_nc["PathY1"]

DevDist000_nc = mat_nc["DevDist000"]
DevAng000_nc = mat_nc["DevAng000"]
DevDist005_nc = mat_nc["DevDist005"]
DevAng005_nc = mat_nc["DevAng005"]
DevDist010_nc = mat_nc["DevDist010"]
DevAng010_nc = mat_nc["DevAng010"]
DevDist015_nc = mat_nc["DevDist015"]
DevAng015_nc = mat_nc["DevAng015"]
DevDist020_nc = mat_nc["DevDist020"]
DevAng020_nc = mat_nc["DevAng020"]
DevDist025_nc = mat_nc["DevDist025"]
DevAng025_nc = mat_nc["DevAng025"]
DevDist030_nc = mat_nc["DevDist030"]
DevAng030_nc = mat_nc["DevAng030"]
DevDist035_nc = mat_nc["DevDist035"]
DevAng035_nc = mat_nc["DevAng035"]
DevDist040_nc = mat_nc["DevDist040"]
DevAng040_nc = mat_nc["DevAng040"]
DevDist045_nc = mat_nc["DevDist045"]
DevAng045_nc = mat_nc["DevAng045"]
DevDist050_nc = mat_nc["DevDist050"]
DevAng050_nc = mat_nc["DevAng050"]
DevDist055_nc = mat_nc["DevDist055"]
DevAng055_nc = mat_nc["DevAng055"]
DevDist060_nc = mat_nc["DevDist050"]
DevAng060_nc = mat_nc["DevAng060"]
DevDist065_nc = mat_nc["DevDist065"]
DevAng065_nc = mat_nc["DevAng065"]
DevDist070_nc = mat_nc["DevDist070"]
DevAng070_nc = mat_nc["DevAng070"]
DevDist075_nc = mat_nc["DevDist075"]
DevAng075_nc = mat_nc["DevAng075"]
DevDist080_nc = mat_nc["DevDist080"]
DevAng080_nc = mat_nc["DevAng080"]
DevDist085_nc = mat_nc["DevDist085"]
DevAng085_nc = mat_nc["DevAng085"]
DevDist090_nc = mat_nc["DevDist090"]
DevAng090_nc = mat_nc["DevAng090"]
DevDist095_nc = mat_nc["DevDist095"]
DevAng095_nc = mat_nc["DevAng095"]
DevDist100_nc = mat_nc["DevDist100"]
DevAng100_nc = mat_nc["DevAng100"]
DevDist105_nc = mat_nc["DevDist105"]
DevAng105_nc = mat_nc["DevAng105"]
DevDist110_nc = mat_nc["DevDist110"]
DevAng110_nc = mat_nc["DevAng110"]
DevDist115_nc = mat_nc["DevDist115"]
DevAng115_nc = mat_nc["DevAng115"]
DevDist120_nc = mat_nc["DevDist120"]
DevAng120_nc = mat_nc["DevAng120"]
DevDist125_nc = mat_nc["DevDist125"]
DevAng125_nc = mat_nc["DevAng125"]
DevDist130_nc = mat_nc["DevDist130"]
DevAng130_nc = mat_nc["DevAng130"]
DevDist135_nc = mat_nc["DevDist135"]
DevAng135_nc = mat_nc["DevAng135"]
DevDist140_nc = mat_nc["DevDist140"]
DevAng140_nc = mat_nc["DevAng140"]
DevDist145_nc = mat_nc["DevDist145"]
DevAng145_nc = mat_nc["DevAng145"]
DevDist150_nc = mat_nc["DevDist150"]
DevAng150_nc = mat_nc["DevAng150"]
DevDist155_nc = mat_nc["DevDist155"]
DevAng155_nc = mat_nc["DevAng155"]
DevDist160_nc = mat_nc["DevDist150"]
DevAng160_nc = mat_nc["DevAng160"]
DevDist165_nc = mat_nc["DevDist165"]
DevAng165_nc = mat_nc["DevAng165"]
DevDist170_nc = mat_nc["DevDist170"]
DevAng170_nc = mat_nc["DevAng170"]
DevDist175_nc = mat_nc["DevDist175"]
DevAng175_nc = mat_nc["DevAng175"]
DevDist180_nc = mat_nc["DevDist180"]
DevAng180_nc = mat_nc["DevAng180"]
DevDist185_nc = mat_nc["DevDist185"]
DevAng185_nc = mat_nc["DevAng185"]
DevDist190_nc = mat_nc["DevDist190"]
DevAng190_nc = mat_nc["DevAng190"]
DevDist195_nc = mat_nc["DevDist195"]
DevAng195_nc = mat_nc["DevAng195"]


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

plt.figure(6)
plt.title("DevDist000 005 010 015 cc")
plt.plot(DevDist000_nc, 'blue', label='00')
plt.plot(DevDist005_nc, 'red', label='05')
plt.plot(DevDist010_nc, 'orange', label='10')
plt.plot(DevDist015_nc, 'green', label='15')
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
plt.plot(DevAng000_cc, 'blue', label='00')
plt.plot(DevAng005_cc, 'red', label='05')
plt.plot(DevAng010_cc, 'orange', label='10')
plt.plot(DevAng015_cc, 'green', label='15')
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

plt.figure(24)
plt.title("path 0, 1")
plt.plot(PathX_nc, PathY_nc, 'blue')
plt.plot(PathX1_nc, PathY1_nc, 'red')
plt.show()

plt.figure(25)
s = 10
v = float(Vel[300])
t = int(s / v * 1000)
print(t)
plt.plot(PathX_nc[300 : 300 + 3600], PathY_nc[300 : 300 + 3600], "blue", label="path")
plt.plot(Pos_X_nc[300], Pos_Y_nc[300], "blue", marker='o', label="pos")
x, y = rotate((Pos_X_nc[300], Pos_Y_nc[300]), Ori[300])
plt.plot(x, y, "lightblue", marker='o', label="sens")
xw, yw = rotate((x, y), Ori[300] + WheAng_nc[300])
plt.plot(xw, yw, "black", marker='o', label="whe ang")
xp, yp = PathX_nc[300 + t], PathY_nc[300 + t]
plt.plot(xp, yp, "brown", marker='o', label="prev p")
xp1, yp1 = PathX1_nc[300], PathY1_nc[300]
plt.plot(xp1, yp1, "green", marker='o', label="prev p1")
plt.axis('equal')
plt.legend(loc=(1.04, 1))
plt.show()

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
