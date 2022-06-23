"""
@author: Bagoly Zoltán
"""

import scipy.io
import matplotlib.pyplot as plt
import numpy as np

#nc = scipy.io.loadmat("meas_cornercut03_speedcontrol5__X_Y_Ori_Vel_AngVel_WFs_SW_DevAng00051015_DevDist00051015_TO_PATH_RouteXYZ_Path_XYZ.mat")
nc = scipy.io.loadmat("meas_cornercut0_speedcontrol20__X_Y_Ori_Vel_AngVel_WFs_WheAngFLFR_DevAng00051015_DevDist00051015_TO_PATH_RouteXYZ_Path_XYZ.mat")

DEV_DIST_GRID = 12
DEV_ANG_GRID = 12
VEL_GRID = 3

# 4: min, max, avg, std
data_grid_5 = np.zeros([DEV_DIST_GRID, DEV_ANG_GRID, 4], dtype=float)
data_grid_5_vel = np.zeros([DEV_DIST_GRID, DEV_ANG_GRID, VEL_GRID, 4], dtype=float)
data_grid_10 = np.zeros([DEV_DIST_GRID, DEV_ANG_GRID, 4], dtype=float)
data_grid_10_vel = np.zeros([DEV_DIST_GRID, DEV_ANG_GRID, VEL_GRID, 4], dtype=float)
data_grid_15 = np.zeros([DEV_DIST_GRID, DEV_ANG_GRID, 4], dtype=float)
data_grid_15_vel = np.zeros([DEV_DIST_GRID, DEV_ANG_GRID, VEL_GRID, 4], dtype=float)
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


colours = ['black', 'blue', 'green', 'brown', 'red', 'cyan', 'magenta', 
           'yellow', 'darkblue', 'orange', 'pink', 'beige', 'coral', 'crimson', 
           'darkgreen', 'fuchsia', 'goldenrod', 'grey', 'yellowgreen', 'lavender', 
           'lightblue', 'lime', 'navy', 'sienna', 'silver',
           'orchid', 'wheat', 'white', 'chocolate', 'khaki', 'azure',
           'salmon', 'plum']
styles = ['-', '--', '-.', ':', 'solid']

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
Vel_min = Vel_nc.min()
Vel_max = Vel_nc.max()
edges_vel = [None] * (VEL_GRID - 1)
grid_vel_size = (Vel_max - Vel_min) / DEV_ANG_GRID
last_edge = Vel_min
for i in range(len(edges_vel)):
    edges_vel[i] = last_edge + grid_vel_size
    last_edge = edges_vel[i]
# VelX_nc = nc["VelX"]
# VelY_nc = nc["VelY"]

WheAng = nc["WheAng_avg"]
#WheAng = nc["SteAng"]

# plt.figure(20)
# plt.title("Vel s")
# plt.plot(Vel_nc, 'blue', label="Vel")
# plt.plot(VelX_nc, 'red', label="X")
# plt.plot(VelY_nc, 'green', label="Y")
# plt.legend(loc=1)
# plt.show()


for i in range(len(WheAng)):
    dist_idx = 0
    ang_idx = 0
    for edge_dist in edges_5_dist:
        if edge_dist < DevDist05nc[i, 0]:
            dist_idx += 1
    for edge_ang in edges_5_ang:
        if edge_ang < DevAng05nc[i, 0]:
            ang_idx += 1
    wheel_angles_5[dist_idx][ang_idx].append(WheAng[i, 0])
for d in range(DEV_DIST_GRID):
    for a in range(DEV_ANG_GRID):
        wheel_angles_5[d][a].sort()
        if wheel_angles_5[d][a] != []:
            data_grid_5[d, a, 0] = wheel_angles_5[d][a][0]
            data_grid_5[d, a, 1] = wheel_angles_5[d][a][-1]
            data_grid_5[d, a, 2] = sum(wheel_angles_5[d][a]) / len(wheel_angles_5[d][a])
            data_grid_5[d, a, 3] = np.std(wheel_angles_5[d][a])
        
for i in range(len(WheAng)):
    dist_idx = 0
    ang_idx = 0
    for edge_dist in edges_10_dist:
        if edge_dist < DevDist10nc[i, 0]:
            dist_idx += 1
    for edge_ang in edges_10_ang:
        if edge_ang < DevAng10nc[i, 0]:
            ang_idx += 1
    wheel_angles_10[dist_idx][ang_idx].append(WheAng[i, 0])
for d in range(DEV_DIST_GRID):
    for a in range(DEV_ANG_GRID):
        wheel_angles_10[d][a].sort()
        if wheel_angles_10[d][a] != []:
            data_grid_10[d, a, 0] = wheel_angles_10[d][a][0]
            data_grid_10[d, a, 1] = wheel_angles_10[d][a][-1]
            data_grid_10[d, a, 2] = sum(wheel_angles_10[d][a]) / len(wheel_angles_10[d][a])
            data_grid_10[d, a, 3] = np.std(wheel_angles_10[d][a])
        
for i in range(len(WheAng)):
    dist_idx = 0
    ang_idx = 0
    for edge_dist in edges_15_dist:
        if edge_dist < DevDist15nc[i, 0]:
            dist_idx += 1
    for edge_ang in edges_15_ang:
        if edge_ang < DevAng15nc[i, 0]:
            ang_idx += 1
    wheel_angles_15[dist_idx][ang_idx].append(WheAng[i, 0])
for d in range(DEV_DIST_GRID):
    for a in range(DEV_ANG_GRID):
        wheel_angles_15[d][a].sort()
        if wheel_angles_15[d][a] != []:
            data_grid_15[d, a, 0] = wheel_angles_15[d][a][0]
            data_grid_15[d, a, 1] = wheel_angles_15[d][a][-1]
            data_grid_15[d, a, 2] = sum(wheel_angles_15[d][a]) / len(wheel_angles_15[d][a])
            data_grid_15[d, a, 3] = np.std(wheel_angles_15[d][a])
        
for i in range(200, len(WheAng)):
    dist_idx = 0
    ang_idx = 0
    vel_idx = 0
    for edge_dist in edges_5_dist:
        if edge_dist < DevDist05nc[i, 0]:
            dist_idx += 1
    for edge_ang in edges_5_ang:
        if edge_ang < DevAng05nc[i, 0]:
            ang_idx += 1
    for edge_vel in edges_vel:
        if edge_vel < Vel_nc[i, 0]:
            vel_idx += 1
    wheel_angles_5_vel[dist_idx][ang_idx][vel_idx].append(WheAng[i, 0])
for d in range(DEV_DIST_GRID):
    for a in range(DEV_ANG_GRID):
        for v in range(VEL_GRID):
            wheel_angles_5_vel[d][a][v].sort()
            if wheel_angles_5_vel[d][a][v] != []:
                data_grid_5_vel[d, a, v, 0] = wheel_angles_5_vel[d][a][v][0]
                data_grid_5_vel[d, a, v, 1] = wheel_angles_5_vel[d][a][v][-1]
                data_grid_5_vel[d, a, v, 2] = sum(wheel_angles_5_vel[d][a][v]) / len(wheel_angles_5_vel[d][a][v])
                data_grid_5_vel[d, a, v, 3] = np.std(wheel_angles_5_vel[d][a][v])
        
for i in range(len(WheAng)):
    dist_idx = 0
    ang_idx = 0
    vel_idx = 0
    for edge_dist in edges_10_dist:
        if edge_dist < DevDist10nc[i, 0]:
            dist_idx += 1
    for edge_ang in edges_10_ang:
        if edge_ang < DevAng10nc[i, 0]:
            ang_idx += 1
    for edge_vel in edges_vel:
        if edge_vel < Vel_nc[i, 0]:
            vel_idx += 1
    wheel_angles_10_vel[dist_idx][ang_idx][vel_idx].append(WheAng[i, 0])
for d in range(DEV_DIST_GRID):
    for a in range(DEV_ANG_GRID):
        for v in range(VEL_GRID):
            wheel_angles_10_vel[d][a][v].sort()
            if wheel_angles_10_vel[d][a][v] != []:
                data_grid_10_vel[d, a, v, 0] = wheel_angles_10_vel[d][a][v][0]
                data_grid_10_vel[d, a, v, 1] = wheel_angles_10_vel[d][a][v][-1]
                data_grid_10_vel[d, a, v, 2] = sum(wheel_angles_10_vel[d][a][v]) / len(wheel_angles_10_vel[d][a][v])
                data_grid_10_vel[d, a, v, 3] = np.std(wheel_angles_10_vel[d][a][v])
        
for i in range(len(WheAng)):
    dist_idx = 0
    ang_idx = 0
    vel_idx = 0
    for edge_dist in edges_15_dist:
        if edge_dist < DevDist15nc[i, 0]:
            dist_idx += 1
    for edge_ang in edges_15_ang:
        if edge_ang < DevAng15nc[i, 0]:
            ang_idx += 1
    for edge_vel in edges_vel:
        if edge_vel < Vel_nc[i, 0]:
            vel_idx += 1
    wheel_angles_15_vel[dist_idx][ang_idx][vel_idx].append(WheAng[i, 0])
for d in range(DEV_DIST_GRID):
    for a in range(DEV_ANG_GRID):
        for v in range(VEL_GRID):
            wheel_angles_15_vel[d][a][v].sort()
            if wheel_angles_15_vel[d][a][v] != []:
                data_grid_15_vel[d, a, v, 0] = wheel_angles_15_vel[d][a][v][0]
                data_grid_15_vel[d, a, v, 1] = wheel_angles_15_vel[d][a][v][-1]
                data_grid_15_vel[d, a, v, 2] = sum(wheel_angles_15_vel[d][a][v]) / len(wheel_angles_15_vel[d][a][v])
                data_grid_15_vel[d, a, v, 3] = np.std(wheel_angles_15_vel[d][a][v])

data_grid_5_vel_std_nc = data_grid_5_vel[:, :, :, 3]
data_grid_10_vel_std_nc = data_grid_10_vel[:, :, :, 3]
data_grid_15_vel_std_nc = data_grid_15_vel[:, :, :, 3]

fig_num = 0
for d in range(DEV_DIST_GRID):
    for a in range(DEV_ANG_GRID):
        plt.figure(fig_num)
        plt.title('road sen 5 m: ' + 
                  'dist ' + str(d+1) + '/' + str(DEV_DIST_GRID) + 
                  '; ang ' + str(a+1) + '/' + str(DEV_ANG_GRID) + '\n' +
                  'min ' + str(data_grid_5[d, a, 0]) + 
                  '; max ' + str(data_grid_5[d, a, 1]) + '\n' +
                  'avg ' + str(data_grid_5[d, a, 2]) + 
                  '; std ' + str(data_grid_5[d, a, 3]) + '\n'
                  )

        plt.xlabel('data points')
        plt.ylabel('wheel angles')
        plt.plot(wheel_angles_5[d][a], 'red')
        if wheel_angles_5[d][a] != []:
            plt.axis([-2, len(wheel_angles_5[d][a]) + 1, -0.5, 0.5])
        plt.show()
        fig_num += 1

for d in range(DEV_DIST_GRID):
    for a in range(DEV_ANG_GRID):
        plt.figure(fig_num)
        plt.title('road sen 10 m: ' + 
                  'dist ' + str(d+1) + '/' + str(DEV_DIST_GRID) + 
                  '; ang ' + str(a+1) + '/' + str(DEV_ANG_GRID) + '\n' +
                  'min ' + str(data_grid_10[d, a, 0]) + 
                  '; max ' + str(data_grid_10[d, a, 1]) + '\n' +
                  'avg ' + str(data_grid_10[d, a, 2]) + 
                  '; std ' + str(data_grid_10[d, a, 3]) + '\n'
                  )

        plt.xlabel('data points')
        plt.ylabel('wheel angles')
        plt.plot(wheel_angles_10[d][a], 'orange')
        if wheel_angles_10[d][a] != []:
            plt.axis([-2, len(wheel_angles_10[d][a]) + 1, -0.5, 0.5])
        plt.show()
        fig_num += 1

for d in range(DEV_DIST_GRID):
    for a in range(DEV_ANG_GRID):
        plt.figure(fig_num)
        plt.title('road sen 15 m: ' + 
                  'dist ' + str(d+1) + '/' + str(DEV_DIST_GRID) + 
                  '; ang ' + str(a+1) + '/' + str(DEV_ANG_GRID) + '\n' +
                  'min ' + str(data_grid_15[d, a, 0]) + 
                  '; max ' + str(data_grid_15[d, a, 1]) + '\n' +
                  'avg ' + str(data_grid_15[d, a, 2]) + 
                  '; std ' + str(data_grid_15[d, a, 3]) + '\n'
                  )

        plt.xlabel('data points')
        plt.ylabel('wheel angles')
        plt.plot(wheel_angles_15[d][a], 'green')
        if wheel_angles_15[d][a] != []:
            plt.axis([-2, len(wheel_angles_15[d][a]) + 1, -0.5, 0.5])
        plt.show()
        fig_num += 1

for d in range(DEV_DIST_GRID):
    for a in range(DEV_ANG_GRID):
        for v in range(VEL_GRID):
            plt.figure(fig_num)
            plt.title('road sen 5 m: ' + 
                      'dist ' + str(d+1) + '/' + str(DEV_DIST_GRID) + 
                      '; ang ' + str(a+1) + '/' + str(DEV_ANG_GRID) + 
                      '; vel ' + str(v+1) + '/' + str(VEL_GRID) + '\n' +
                      'min ' + str(data_grid_5_vel[d, a, v, 0]) + 
                      '; max ' + str(data_grid_5_vel[d, a, v, 1]) + '\n' +
                      'avg ' + str(data_grid_5_vel[d, a, v, 2]) + 
                      '; std ' + str(data_grid_5_vel[d, a, v, 3]) + '\n'
                      )
    
            plt.xlabel('data points')
            plt.ylabel('wheel angles')
            plt.plot(wheel_angles_5_vel[d][a][v], 'red')
            if wheel_angles_5_vel[d][a][v] != []:
                plt.axis([-2, len(wheel_angles_5_vel[d][a][v]) + 1, -0.5, 0.5])
            plt.show()
            fig_num += 1

for d in range(DEV_DIST_GRID):
    for a in range(DEV_ANG_GRID):
        for v in range(VEL_GRID):
            plt.figure(fig_num)
            plt.title('road sen 10 m: ' + 
                      'dist ' + str(d+1) + '/' + str(DEV_DIST_GRID) + 
                      '; ang ' + str(a+1) + '/' + str(DEV_ANG_GRID) + 
                      '; vel ' + str(v+1) + '/' + str(VEL_GRID) + '\n' +
                      'min ' + str(data_grid_10_vel[d, a, v, 0]) + 
                      '; max ' + str(data_grid_10_vel[d, a, v, 1]) + '\n' +
                      'avg ' + str(data_grid_10_vel[d, a, v, 2]) + 
                      '; std ' + str(data_grid_10_vel[d, a, v, 3]) + '\n'
                      )
    
            plt.xlabel('data points')
            plt.ylabel('wheel angles')
            plt.plot(wheel_angles_10_vel[d][a][v], 'orange')
            if wheel_angles_10_vel[d][a][v] != []:
                plt.axis([-2, len(wheel_angles_10_vel[d][a][v]) + 1, -0.5, 0.5])
            plt.show()
            fig_num += 1

for d in range(DEV_DIST_GRID):
    for a in range(DEV_ANG_GRID):
        for v in range(VEL_GRID):
            plt.figure(fig_num)
            plt.title('road sen 15 m: ' + 
                      'dist ' + str(d+1) + '/' + str(DEV_DIST_GRID) + 
                      '; ang ' + str(a+1) + '/' + str(DEV_ANG_GRID) + 
                      '; vel ' + str(v+1) + '/' + str(VEL_GRID) + '\n' +
                      'min ' + str(data_grid_15_vel[d, a, v, 0]) + 
                      '; max ' + str(data_grid_15_vel[d, a, v, 1]) + '\n' +
                      'avg ' + str(data_grid_15_vel[d, a, v, 2]) + 
                      '; std ' + str(data_grid_15_vel[d, a, v, 3]) + '\n'
                      )
    
            plt.xlabel('data points')
            plt.ylabel('wheel angles')
            plt.plot(wheel_angles_15_vel[d][a][v], 'green')
            if wheel_angles_15_vel[d][a][v] != []:
                plt.axis([-2, len(wheel_angles_15_vel[d][a][v]) + 1, -0.5, 0.5])
            plt.show()
            fig_num += 1
