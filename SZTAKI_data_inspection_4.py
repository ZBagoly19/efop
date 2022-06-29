"""
@author: Bagoly Zoltán
"""

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm


WHE_ANG_RANGE = 0.3

#nc = scipy.io.loadmat("meas_cornercut03_speedcontrol25_TS0001__X_Y_Ori_Vel_AngVel_WFs_WheAngFLFR_DevAng00051015_DevDist00051015_TO_PATH_RouteXYZ_Path_XYZ.mat")
nc = scipy.io.loadmat("meas_cornercut0_speedcontrol25_TS0001__X_Y_Ori_Vel_AngVel_WFs_WheAngFLFR_DevAng00051015_DevDist00051015_TO_PATH_RouteXYZ_Path_XYZ.mat")

DEV_DIST_GRID = 24
DEV_ANG_GRID = 60
VEL_GRID = 4

# 4: min, max, avg, std
data_grid_0 = np.zeros([DEV_DIST_GRID, DEV_ANG_GRID, 4], dtype=float)
data_grid_0_vel = np.zeros([DEV_DIST_GRID, DEV_ANG_GRID, VEL_GRID, 4], dtype=float)
data_grid_5 = np.zeros([DEV_DIST_GRID, DEV_ANG_GRID, 4], dtype=float)
data_grid_5_vel = np.zeros([DEV_DIST_GRID, DEV_ANG_GRID, VEL_GRID, 4], dtype=float)
data_grid_10 = np.zeros([DEV_DIST_GRID, DEV_ANG_GRID, 4], dtype=float)
data_grid_10_vel = np.zeros([DEV_DIST_GRID, DEV_ANG_GRID, VEL_GRID, 4], dtype=float)
data_grid_15 = np.zeros([DEV_DIST_GRID, DEV_ANG_GRID, 4], dtype=float)
data_grid_15_vel = np.zeros([DEV_DIST_GRID, DEV_ANG_GRID, VEL_GRID, 4], dtype=float)

wheel_angles_0 = [None] * DEV_DIST_GRID
for i in range(DEV_DIST_GRID):
    wheel_angles_0[i] = [None] * DEV_ANG_GRID
for i in range(DEV_DIST_GRID):
    for j in range(DEV_ANG_GRID):
        wheel_angles_0[i][j] = []

wheel_angles_5 = [None] * DEV_DIST_GRID
for i in range(DEV_DIST_GRID):
    wheel_angles_5[i] = [None] * DEV_ANG_GRID
for i in range(DEV_DIST_GRID):
    for j in range(DEV_ANG_GRID):
        wheel_angles_5[i][j] = []

wheel_angles_10 = [None] * DEV_DIST_GRID
for i in range(DEV_DIST_GRID):
    wheel_angles_10[i] = [None] * DEV_ANG_GRID
for i in range(DEV_DIST_GRID):
    for j in range(DEV_ANG_GRID):
        wheel_angles_10[i][j] = []

wheel_angles_15 = [None] * DEV_DIST_GRID
for i in range(DEV_DIST_GRID):
    wheel_angles_15[i] = [None] * DEV_ANG_GRID
for i in range(DEV_DIST_GRID):
    for j in range(DEV_ANG_GRID):
        wheel_angles_15[i][j] = []
        
wheel_angles_0_vel = [None] * DEV_DIST_GRID
for i in range(DEV_DIST_GRID):
    wheel_angles_0_vel[i] = [None] * DEV_ANG_GRID
for i in range(DEV_DIST_GRID):
    for j in range(DEV_ANG_GRID):
        wheel_angles_0_vel[i][j] = [None] * VEL_GRID
for i in range(DEV_DIST_GRID):
    for j in range(DEV_ANG_GRID):
        for k in range(VEL_GRID):
            wheel_angles_0_vel[i][j][k] = []

wheel_angles_5_vel = [None] * DEV_DIST_GRID
for i in range(DEV_DIST_GRID):
    wheel_angles_5_vel[i] = [None] * DEV_ANG_GRID
for i in range(DEV_DIST_GRID):
    for j in range(DEV_ANG_GRID):
        wheel_angles_5_vel[i][j] = [None] * VEL_GRID
for i in range(DEV_DIST_GRID):
    for j in range(DEV_ANG_GRID):
        for k in range(VEL_GRID):
            wheel_angles_5_vel[i][j][k] = []

wheel_angles_10_vel = [None] * DEV_DIST_GRID
for i in range(DEV_DIST_GRID):
    wheel_angles_10_vel[i] = [None] * DEV_ANG_GRID
for i in range(DEV_DIST_GRID):
    for j in range(DEV_ANG_GRID):
        wheel_angles_10_vel[i][j] = [None] * VEL_GRID
for i in range(DEV_DIST_GRID):
    for j in range(DEV_ANG_GRID):
        for k in range(VEL_GRID):
            wheel_angles_10_vel[i][j][k] = []
            
wheel_angles_15_vel = [None] * DEV_DIST_GRID
for i in range(DEV_DIST_GRID):
    wheel_angles_15_vel[i] = [None] * DEV_ANG_GRID
for i in range(DEV_DIST_GRID):
    for j in range(DEV_ANG_GRID):
        wheel_angles_15_vel[i][j] = [None] * VEL_GRID
for i in range(DEV_DIST_GRID):
    for j in range(DEV_ANG_GRID):
        for k in range(VEL_GRID):
            wheel_angles_15_vel[i][j][k] = []

DevDist00nc = nc["DevDist00"]
DevDist00nc_min = DevDist00nc.min()
DevDist00nc_max = DevDist00nc.max()
edges_0_dist = [None] * (DEV_DIST_GRID - 1)
grid_0_dist_size = (DevDist00nc_max - DevDist00nc_min) / DEV_DIST_GRID
last_edge = DevDist00nc_min
for i in range(len(edges_0_dist)):
    edges_0_dist[i] = last_edge + grid_0_dist_size
    last_edge = edges_0_dist[i]

DevAng00nc = nc["DevAng00"]
DevAng00nc_min = DevAng00nc.min()
DevAng00nc_max = DevAng00nc.max()
edges_0_ang = [None] * (DEV_ANG_GRID - 1)
grid_0_ang_size = (DevAng00nc_max - DevAng00nc_min) / DEV_ANG_GRID
last_edge = DevAng00nc_min
for i in range(len(edges_0_ang)):
    edges_0_ang[i] = last_edge + grid_0_ang_size
    last_edge = edges_0_ang[i]

DevDist05nc = nc["DevDist05"]
DevDist05nc_min = DevDist05nc.min()
DevDist05nc_max = DevDist05nc.max()
edges_5_dist = [None] * (DEV_DIST_GRID - 1)
grid_5_dist_size = (DevDist05nc_max - DevDist05nc_min) / DEV_DIST_GRID
last_edge = DevDist05nc_min
for i in range(len(edges_5_dist)):
    edges_5_dist[i] = last_edge + grid_5_dist_size
    last_edge = edges_5_dist[i]

DevAng05nc = nc["DevAng05"]
DevAng05nc_min = DevAng05nc.min()
DevAng05nc_max = DevAng05nc.max()
edges_5_ang = [None] * (DEV_ANG_GRID - 1)
grid_5_ang_size = (DevAng05nc_max - DevAng05nc_min) / DEV_ANG_GRID
last_edge = DevAng05nc_min
for i in range(len(edges_5_ang)):
    edges_5_ang[i] = last_edge + grid_5_ang_size
    last_edge = edges_5_ang[i]

DevDist10nc = nc["DevDist10"]
DevDist10nc_min = DevDist10nc.min()
DevDist10nc_max = DevDist10nc.max()
edges_10_dist = [None] * (DEV_DIST_GRID - 1)
grid_10_dist_size = (DevDist10nc_max - DevDist10nc_min) / DEV_DIST_GRID
last_edge = DevDist10nc_min
for i in range(len(edges_10_dist)):
    edges_10_dist[i] = last_edge + grid_10_dist_size
    last_edge = edges_10_dist[i]

DevAng10nc = nc["DevAng10"]
DevAng10nc_min = DevAng10nc.min()
DevAng10nc_max = DevAng10nc.max()
edges_10_ang = [None] * (DEV_ANG_GRID - 1)
grid_10_ang_size = (DevAng10nc_max - DevAng10nc_min) / DEV_ANG_GRID
last_edge = DevAng10nc_min
for i in range(len(edges_10_ang)):
    edges_10_ang[i] = last_edge + grid_10_ang_size
    last_edge = edges_10_ang[i]

DevDist15nc = nc["DevDist15"]
DevDist15nc_min = DevDist15nc.min()
DevDist15nc_max = DevDist15nc.max()
edges_15_dist = [None] * (DEV_DIST_GRID - 1)
grid_15_dist_size = (DevDist15nc_max - DevDist15nc_min) / DEV_DIST_GRID
last_edge = DevDist15nc_min
for i in range(len(edges_15_dist)):
    edges_15_dist[i] = last_edge + grid_15_dist_size
    last_edge = edges_15_dist[i]

DevAng15nc = nc["DevAng15"]
DevAng15nc_min = DevAng15nc.min()
DevAng15nc_max = DevAng15nc.max()
edges_15_ang = [None] * (DEV_ANG_GRID - 1)
grid_15_ang_size = (DevAng15nc_max - DevAng15nc_min) / DEV_ANG_GRID
last_edge = DevAng15nc_min
for i in range(len(edges_15_ang)):
    edges_15_ang[i] = last_edge + grid_15_ang_size
    last_edge = edges_15_ang[i]

Vel_nc = nc["Vel"]
plt.plot(Vel_nc)
Vel_min = Vel_nc.min()
Vel_max = Vel_nc.max()
edges_vel = [None] * (VEL_GRID - 1)
grid_vel_size = (Vel_max - Vel_min) / VEL_GRID
last_edge = Vel_min
for i in range(len(edges_vel)):
    edges_vel[i] = last_edge + grid_vel_size
    last_edge = edges_vel[i]
# VelX_nc = nc["VelX"]
# VelY_nc = nc["VelY"]

WheAng = nc["WheAng_avg"]

# plt.figure(20)
# plt.title("Vel s")
# plt.plot(Vel_nc, 'blue', label="Vel")
# plt.plot(VelX_nc, 'red', label="X")
# plt.plot(VelY_nc, 'green', label="Y")
# plt.legend(loc=(1.04, 1))
# plt.show()

def wheel_angles(sensor):
    if sensor == 0:
        edges_dist = edges_0_dist
        edges_ang = edges_0_ang
        data_dist = DevDist00nc
        data_ang = DevAng00nc
        grid = data_grid_0
        angles = wheel_angles_0
    elif sensor == 5:
        edges_dist = edges_5_dist
        edges_ang = edges_5_ang
        data_dist = DevDist05nc
        data_ang = DevAng05nc
        grid = data_grid_5
        angles = wheel_angles_5
    elif sensor == 10:
        edges_dist = edges_10_dist
        edges_ang = edges_10_ang
        data_dist = DevDist10nc
        data_ang = DevAng10nc
        grid = data_grid_10
        angles = wheel_angles_10
    elif sensor == 15:
        edges_dist = edges_15_dist
        edges_ang = edges_15_ang
        data_dist = DevDist15nc
        data_ang = DevAng15nc
        grid = data_grid_15
        angles = wheel_angles_15
    else:
        print("Hibás sensor érték!")
    
    for i in tqdm(range(200, len(WheAng) - 200)):
        dist_idx = 0
        ang_idx = 0
  
        for edge_dist in edges_dist:
            if edge_dist < data_dist[i, 0]:
                dist_idx += 1
        for edge_ang in edges_ang:
            if edge_ang < data_ang[i, 0]:
                ang_idx += 1
        angles[dist_idx][ang_idx].append([WheAng[i, 0], i])
        
    for d in range(DEV_DIST_GRID):
        for a in range(DEV_ANG_GRID):
            angles[d][a].sort()
            # if angles[d][a] != []:
            #     grid[d, a, 0] = angles[d][a][0]
            #     grid[d, a, 1] = angles[d][a][-1]
            #     grid[d, a, 2] = sum(angles[d][a]) / len(angles[d][a])
            #     grid[d, a, 3] = np.std(angles[d][a])

sensors = [0, 5, 10, 15]
for s in sensors:
    wheel_angles(s)

def wheel_angles_v(sensor):
    if sensor == 0:
        edges_dist = edges_0_dist
        edges_ang = edges_0_ang
        data_dist = DevDist00nc
        data_ang = DevAng00nc
        grid = data_grid_0_vel
        angles = wheel_angles_0_vel
    elif sensor == 5:
        edges_dist = edges_5_dist
        edges_ang = edges_5_ang
        data_dist = DevDist05nc
        data_ang = DevAng05nc
        grid = data_grid_5_vel
        angles = wheel_angles_5_vel
    elif sensor == 10:
        edges_dist = edges_10_dist
        edges_ang = edges_10_ang
        data_dist = DevDist10nc
        data_ang = DevAng10nc
        grid = data_grid_10_vel
        angles = wheel_angles_10_vel
    elif sensor == 15:
        edges_dist = edges_15_dist
        edges_ang = edges_15_ang
        data_dist = DevDist15nc
        data_ang = DevAng15nc
        grid = data_grid_15_vel
        angles = wheel_angles_15_vel
    else:
        print("Hibás sensor érték!")
        
    for i in tqdm(range(200, len(WheAng))):
        dist_idx = 0
        ang_idx = 0
        vel_idx = 0

        for edge_dist in edges_dist:
            if edge_dist < data_dist[i, 0]:
                dist_idx += 1
        for edge_ang in edges_ang:
            if edge_ang < data_ang[i, 0]:
                ang_idx += 1
        for edge_vel in edges_vel:
            if edge_vel < Vel_nc[i, 0]:
                vel_idx += 1
        angles[dist_idx][ang_idx][vel_idx].append([WheAng[i, 0], i])
        
    for d in range(DEV_DIST_GRID):
        for a in range(DEV_ANG_GRID):
            for v in range(VEL_GRID):
                angles[d][a][v].sort()
                # if angles[d][a][v] != []:
                #     grid[d, a, v, 0] = angles[d][a][v][0]
                #     grid[d, a, v, 1] = angles[d][a][v][-1]
                #     grid[d, a, v, 2] = sum(angles[d][a][v]) / len(angles[d][a][v])
                #     grid[d, a, v, 3] = np.std(angles[d][a][v])

for s in sensors:
    wheel_angles_v(s)

Ori = nc["Ori_Z"]

def rotate(origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    """
    ox, oy = origin
    px, py = ox + 1.45, oy

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

PathX = nc["PathX"]
PathY = nc["PathY"]
Pos_X = nc["Pos_X"]
Pos_Y = nc["Pos_Y"]
def plot_pair(figure_const, sensor):
    if sensor == 0:
        edges_dist = edges_0_dist
        edges_dist.append(DevDist00nc_max)
        edges_dist.append(DevDist00nc_min)
        edges_ang = edges_0_ang
        edges_ang.append(DevAng00nc_max)
        edges_ang.append(DevAng00nc_min)
        data_dist = DevDist00nc
        data_ang = DevAng00nc
        angles = wheel_angles_0
    elif sensor == 5:
        edges_dist = edges_5_dist
        edges_dist.append(DevDist05nc_max)
        edges_dist.append(DevDist05nc_min)
        edges_ang = edges_5_ang
        edges_ang.append(DevAng05nc_max)
        edges_ang.append(DevAng05nc_min)
        data_dist = DevDist05nc
        data_ang = DevAng05nc
        angles = wheel_angles_5
    elif sensor == 10:
        edges_dist = edges_10_dist
        edges_dist.append(DevDist10nc_max)
        edges_dist.append(DevDist10nc_min)
        edges_ang = edges_10_ang
        edges_ang.append(DevAng10nc_max)
        edges_ang.append(DevAng10nc_min)
        data_dist = DevDist10nc
        data_ang = DevAng10nc
        angles = wheel_angles_10
    elif sensor == 15:
        edges_dist = edges_15_dist
        edges_dist.append(DevDist15nc_max)
        edges_dist.append(DevDist15nc_min)
        edges_ang = edges_15_ang
        edges_ang.append(DevAng15nc_max)
        edges_ang.append(DevAng15nc_min)
        data_dist = DevDist15nc
        data_ang = DevAng15nc
        angles = wheel_angles_15
    else:
        print("Hibás sensor érték!")
    
    if sensor == 0:
        edges_dist = edges_0_dist
        edges_ang = edges_0_ang
        data_dist_vel = DevDist00nc
        data_ang_vel = DevAng00nc
        angles_vel = wheel_angles_0_vel
    elif sensor == 5:
        edges_dist = edges_5_dist
        edges_ang = edges_5_ang
        data_dist_vel = DevDist05nc
        data_ang_vel = DevAng05nc
        angles_vel = wheel_angles_5_vel
    elif sensor == 10:
        edges_dist = edges_10_dist
        edges_ang = edges_10_ang
        data_dist_vel = DevDist10nc
        data_ang_vel = DevAng10nc
        angles_vel = wheel_angles_10_vel
    elif sensor == 15:
        edges_dist = edges_15_dist
        edges_ang = edges_15_ang
        data_dist_vel = DevDist15nc
        data_ang_vel = DevAng15nc
        angles_vel = wheel_angles_15_vel
    else:
        print("Hibás sensor érték!")
    edges_v = edges_vel
    edges_v.append(Vel_max)
    edges_v.append(Vel_min)
    
    for d in range(DEV_DIST_GRID):
        for a in range(DEV_ANG_GRID):
            if angles[d][a] != []:
                #print(d, a)
                if WHE_ANG_RANGE < angles[d][a][-1][0] - angles[d][a][0][0]:
                    mom_min = angles[d][a][0][1]
                    mom_max = angles[d][a][-1][1]
                    plt.figure(figure_const)
                    plt.xlabel('')
                    plt.ylabel('')
                    plt.title("sensor: " + str(sensor) + "; moment: " + str(mom_min) + "\n" + 
                              "d: " + str(edges_dist[d - 1]) + " - " + str(edges_dist[d]) + "; a: " + str(edges_ang[a - 1]) + " - " + str(edges_ang[a]) + "\n" + 
                              "dist min: " + str(data_dist[mom_min]) + "; ang min: " + str(data_ang[mom_min]) + "\n" + 
                              "wheel angle min: " + str(WheAng[mom_min]) + "\n" +
                              "dist max: " + str(data_dist[mom_max]) + "; ang max: " + str(data_ang[mom_max]) + "\n" + 
                              "wheel angle max: " + str(WheAng[mom_max]) + "\n"
                              )
                    #plt.title(str(mom_min) + " " + str(mom_max))
                    plt.plot(PathX[mom_min : mom_min + 3600], PathY[mom_min : mom_min + 3600], "red", label="min path")
                    plt.plot(Pos_X[mom_min], Pos_Y[mom_min], "red", marker='o', label="min pos")
                    x, y = rotate((Pos_X[mom_min], Pos_Y[mom_min]), Ori[mom_min])
                    plt.plot(x, y, "orange", marker='o', label="sens")
                    xw, yw = rotate((x, y), Ori[mom_min] + WheAng[mom_min])
                    plt.plot(xw, yw, "black", marker='o', label="whe ang")
                    plt.axis('equal')
                    plt.legend(loc=(1.04, 1))
                    plt.show()
                    plt.figure(figure_const+1)
                    plt.xlabel('')
                    plt.ylabel('')
                    plt.title("sensor: " + str(sensor) + "; moment: " + str(mom_max) + "\n" + 
                              "d: " + str(edges_dist[d - 1]) + " - " + str(edges_dist[d]) + "; a: " + str(edges_ang[a - 1]) + " - " + str(edges_ang[a]) + "\n" + 
                              "dist min: " + str(data_dist[mom_min]) + "; ang min: " + str(data_ang[mom_min]) + "\n" + 
                              "wheel angle min: " + str(WheAng[mom_min]) + "\n" +
                              "dist max: " + str(data_dist[mom_max]) + "; ang max: " + str(data_ang[mom_max]) + "\n" + 
                              "wheel angle max: " + str(WheAng[mom_max]) + "\n"
                              )
                    plt.plot(PathX[mom_max : mom_max + 3600], PathY[mom_max : mom_max + 3600], "blue", label="max path")
                    plt.plot(Pos_X[mom_max], Pos_Y[mom_max], "blue", marker='o', label="max pos")
                    x, y = rotate((Pos_X[mom_max], Pos_Y[mom_max]), Ori[mom_max])
                    plt.plot(x, y, "lightblue", marker='o', label="sens")
                    xw, yw = rotate((x, y), Ori[mom_max] + WheAng[mom_max])
                    plt.plot(xw, yw, "black", marker='o', label="whe ang")
                    plt.axis('equal')
                    plt.legend(loc=(1.04, 1))
                    plt.show()
    for d in range(DEV_DIST_GRID):
        for a in range(DEV_ANG_GRID):
            for v in range(VEL_GRID):
                if angles_vel[d][a][v] != []:
                    #print(d, a)
                    if WHE_ANG_RANGE < angles_vel[d][a][v][-1][0] - angles_vel[d][a][v][0][0]:
                        mom_min = angles_vel[d][a][v][0][1]
                        mom_max = angles_vel[d][a][v][-1][1]
                        plt.figure(figure_const+2)
                        plt.xlabel('')
                        plt.ylabel('')
                        plt.title("sensor: " + str(sensor) + "; moment: " + str(mom_min) + "\n" + 
                                  "d: " + str(edges_dist[d - 1]) + " - " + str(edges_dist[d]) + "; a: " + str(edges_ang[a - 1]) + " - " + str(edges_ang[a]) + "\n" +
                                  "v: " + str(edges_v[v - 1]) + " - " + str(edges_v[v]) + "\n" + 
                                  "dist min: " + str(data_dist_vel[mom_min]) + " ang min: " + str(data_ang_vel[mom_min]) + 
                                  " vel min: " + str(Vel_nc[mom_min]) + "\n" +
                                  "wheel angle min: " + str(WheAng[mom_min]) + "\n" +
                                  "dist max: " + str(data_dist_vel[mom_max]) + " ang max: " + str(data_ang_vel[mom_max]) + 
                                  " vel max: " + str(Vel_nc[mom_max]) + "\n" +
                                  "wheel angle max: " + str(WheAng[mom_max]) + "\n"
                                  )
                        plt.plot(PathX[mom_min : mom_min + 3600], PathY[mom_min : mom_min + 3600], "red", label="min path")
                        plt.plot(Pos_X[mom_min], Pos_Y[mom_min], "red", marker='o', label="min pos")
                        x, y = rotate((Pos_X[mom_min], Pos_Y[mom_min]), Ori[mom_min])
                        plt.plot(x, y, "orange", marker='o', label="sens")
                        xw, yw = rotate((x, y), Ori[mom_min] + WheAng[mom_min])
                        plt.plot(xw, yw, "black", marker='o', label="whe ang")
                        plt.axis('equal')
                        plt.legend(loc=(1.04, 1))
                        plt.show()
                        plt.figure(figure_const+3)
                        plt.xlabel('')
                        plt.ylabel('')
                        plt.title("sensor: " + str(sensor) + "; moment: " + str(mom_max) + "\n" + 
                                  "d: " + str(edges_dist[d - 1]) + " - " + str(edges_dist[d]) + "; a: " + str(edges_ang[a - 1]) + " - " + str(edges_ang[a]) + "\n" +
                                  "v: " + str(edges_v[v - 1]) + " - " + str(edges_v[v]) + "\n" + 
                                  "dist min: " + str(data_dist_vel[mom_min]) + " ang min: " + str(data_ang_vel[mom_min]) + 
                                  " vel min: " + str(Vel_nc[mom_min]) + "\n" +
                                  "wheel angle min: " + str(WheAng[mom_min]) + "\n" +
                                  "dist max: " + str(data_dist_vel[mom_max]) + " ang max: " + str(data_ang_vel[mom_max]) + 
                                  " vel max: " + str(Vel_nc[mom_max]) + "\n" +
                                  "wheel angle max: " + str(WheAng[mom_max]) + "\n"
                                  )
                        plt.plot(PathX[mom_max : mom_max + 3600], PathY[mom_max : mom_max + 3600], "blue", label="max path")
                        plt.plot(Pos_X[mom_max], Pos_Y[mom_max], "blue", marker='o', label="max pos")
                        x, y = rotate((Pos_X[mom_max], Pos_Y[mom_max]), Ori[mom_max])
                        xw, yw = rotate((x, y), Ori[mom_max] + WheAng[mom_max])
                        plt.plot(x, y, "lightblue", marker='o', label="sens")
                        plt.plot(xw, yw, "black", marker='o', label="whe ang")
                        plt.axis('equal')
                        plt.legend(loc=(1.04, 1))
                        plt.show()

fig = 0
for s in sensors:
    plot_pair(fig, s)
    fig += 4
    
plt.plot(PathX[1267275 - 1000 : 1267275 - 1000 + 3600], PathY[1267275 - 1000 : 1267275 - 1000 + 3600], "blue", label="path")
plt.plot(Pos_X[1267275 - 1000], Pos_Y[1267275 - 1000], "blue", marker='o', label="pos")
x, y = rotate((Pos_X[1267275 - 1000], Pos_Y[1267275 - 1000]), Ori[1267275 - 1000])
plt.plot(x, y, "lightblue", marker='o', label="sens")
xw, yw = rotate((x, y), Ori[1267275 - 1000] + WheAng[1267275 - 1000])
plt.plot(xw, yw, "black", marker='o', label="whe ang")
plt.axis('equal')
plt.legend(loc=(1.04, 1))
plt.show()

# plt.plot(WheAng)
