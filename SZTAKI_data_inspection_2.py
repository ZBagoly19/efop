"""
@author: Bagoly Zoltán
"""

import scipy.io
import matplotlib.pyplot as plt
import numpy as np

#nc = scipy.io.loadmat("meas_cornercut03_speedcontrol25_TS0001__X_Y_Ori_Vel_AngVel_WFs_WheAngFLFR_DevAng00051015_DevDist00051015_TO_PATH_RouteXYZ_Path_XYZ.mat")
nc = scipy.io.loadmat("meas_cornercut0_speedcontrol25_TS0001__X_Y_Ori_Vel_AngVel_WFs_WheAngFLFR_DevAng00051015_DevDist00051015_TO_PATH_RouteXYZ_Path_XYZ.mat")

DEV_DIST_GRID = 12
DEV_ANG_GRID = 12
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
# plt.legend(loc=1)
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
    
    for i in range(200, len(WheAng)):
        dist_idx = 0
        ang_idx = 0
  
        for edge_dist in edges_dist:
            if edge_dist < data_dist[i, 0]:
                dist_idx += 1
        for edge_ang in edges_ang:
            if edge_ang < data_ang[i, 0]:
                ang_idx += 1
        angles[dist_idx][ang_idx].append(WheAng[i, 0])
        
    for d in range(DEV_DIST_GRID):
        for a in range(DEV_ANG_GRID):
            angles[d][a].sort()
            if angles[d][a] != []:
                grid[d, a, 0] = angles[d][a][0]
                grid[d, a, 1] = angles[d][a][-1]
                grid[d, a, 2] = sum(angles[d][a]) / len(angles[d][a])
                grid[d, a, 3] = np.std(angles[d][a])

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
        
    for i in range(200, len(WheAng)):
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
        angles[dist_idx][ang_idx][vel_idx].append(WheAng[i, 0])
        
    for d in range(DEV_DIST_GRID):
        for a in range(DEV_ANG_GRID):
            for v in range(VEL_GRID):
                angles[d][a][v].sort()
                if angles[d][a][v] != []:
                    grid[d, a, v, 0] = angles[d][a][v][0]
                    grid[d, a, v, 1] = angles[d][a][v][-1]
                    grid[d, a, v, 2] = sum(angles[d][a][v]) / len(angles[d][a][v])
                    grid[d, a, v, 3] = np.std(angles[d][a][v])

for s in sensors:
    wheel_angles_v(s)

data_grid_5_vel_std_nc = data_grid_5_vel[:, :, :, 3]
data_grid_10_vel_std_nc = data_grid_10_vel[:, :, :, 3]
data_grid_15_vel_std_nc = data_grid_15_vel[:, :, :, 3]


table_2d_all_0_cc = np.zeros([DEV_DIST_GRID * DEV_ANG_GRID, 8], dtype=float)
table_2d_all_5_cc = np.zeros([DEV_DIST_GRID * DEV_ANG_GRID, 8], dtype=float)
table_2d_all_10_cc = np.zeros([DEV_DIST_GRID * DEV_ANG_GRID, 8], dtype=float)
table_2d_all_15_cc = np.zeros([DEV_DIST_GRID * DEV_ANG_GRID, 8], dtype=float)

def fill_table(sensor):
    if sensor == 0:
        grid = data_grid_0
        angles = wheel_angles_0
        table = table_2d_all_0_cc
    elif sensor == 5:
        grid = data_grid_5
        angles = wheel_angles_5
        table = table_2d_all_5_cc
    elif sensor == 10:
        grid = data_grid_10
        angles = wheel_angles_10
        table = table_2d_all_10_cc
    elif sensor == 15:
        grid = data_grid_15
        angles = wheel_angles_15
        table = table_2d_all_15_cc
    else:
        print("Hibás sensor érték!")
        
    table_line = 0
    for d in range(DEV_DIST_GRID):
        for a in range(DEV_ANG_GRID):
            table[table_line][0] = d
            table[table_line][1] = a
            table[table_line][2] = -1
            table[table_line][3] = grid[d, a, 0]
            table[table_line][4] = grid[d, a, 1]
            table[table_line][5] = grid[d, a, 2]
            table[table_line][6] = grid[d, a, 3]
            table[table_line][7] = len(angles[d][a])
            table_line += 1

for s in sensors:
    fill_table(s)

table_2d_all_0_vel_cc = np.zeros([DEV_DIST_GRID * DEV_ANG_GRID * VEL_GRID, 8], dtype=float)
table_2d_all_5_vel_cc = np.zeros([DEV_DIST_GRID * DEV_ANG_GRID * VEL_GRID, 8], dtype=float)
table_2d_all_10_vel_cc = np.zeros([DEV_DIST_GRID * DEV_ANG_GRID * VEL_GRID, 8], dtype=float)
table_2d_all_15_vel_cc = np.zeros([DEV_DIST_GRID * DEV_ANG_GRID * VEL_GRID, 8], dtype=float)

def fill_table(sensor):
    if sensor == 0:
        grid = data_grid_0_vel
        angles = wheel_angles_0_vel
        table = table_2d_all_0_vel_cc
    elif sensor == 5:
        grid = data_grid_5_vel
        angles = wheel_angles_5_vel
        table = table_2d_all_5_vel_cc
    elif sensor == 10:
        grid = data_grid_10_vel
        angles = wheel_angles_10_vel
        table = table_2d_all_10_vel_cc
    elif sensor == 15:
        grid = data_grid_15_vel
        angles = wheel_angles_15_vel
        table = table_2d_all_15_vel_cc
    else:
        print("Hibás sensor érték!")
        
    table_line = 0
    for d in range(DEV_DIST_GRID):
        for a in range(DEV_ANG_GRID):
            for v in range(VEL_GRID):
                table[table_line][0] = d
                table[table_line][1] = a
                table[table_line][2] = v
                table[table_line][3] = grid[d, a, v, 0]
                table[table_line][4] = grid[d, a, v, 1]
                table[table_line][5] = grid[d, a, v, 2]
                table[table_line][6] = grid[d, a, v, 3]
                table[table_line][7] = len(angles[d][a][v])
                table_line += 1

for s in sensors:
    fill_table(s)


def plot_group(sensor, d, a, fig_num):
    if sensor == 0:
        grid = data_grid_0
        angles = wheel_angles_0
        colour = 'black'
    elif sensor == 5:
        grid = data_grid_5
        angles = wheel_angles_5
        colour = 'red'
    elif sensor == 10:
        grid = data_grid_10
        angles = wheel_angles_10
        colour = 'orange'
    elif sensor == 15:
        grid = data_grid_15
        angles = wheel_angles_15
        colour = 'green'
    else:
        print("Hibás sensor érték!")
    plt.figure(fig_num)
    plt.title('road sen ' + str(sensor) + ' m: ' + 
              'dist ' + str(d+1) + '/' + str(DEV_DIST_GRID) + 
              '; ang ' + str(a+1) + '/' + str(DEV_ANG_GRID) + '\n' +
              'min ' + str(grid[d, a, 0]) + 
              '; max ' + str(grid[d, a, 1]) + '\n' +
              'avg ' + str(grid[d, a, 2]) + 
              '; std ' + str(grid[d, a, 3]) + '\n'
              )

    plt.xlabel('data points')
    plt.ylabel('wheel angles')
    plt.plot(angles[d][a], colour)
    if angles[d][a] != []:
        plt.axis([-2, len(angles[d][a]) + 1, -0.5, 0.5])
    plt.show()
    
def plot_group_v(sensor, d, a, v, fig_num):
    if sensor == 0:
        grid = data_grid_0_vel
        angles = wheel_angles_0_vel
        colour = 'black'
    elif sensor == 5:
        grid = data_grid_5_vel
        angles = wheel_angles_5_vel
        colour = 'red'
    elif sensor == 10:
        grid = data_grid_10_vel
        angles = wheel_angles_10_vel
        colour = 'orange'
    elif sensor == 15:
        grid = data_grid_15_vel
        angles = wheel_angles_15_vel
        colour = 'green'
    else:
        print("Hibás sensor érték!")
    plt.figure(fig_num)
    plt.title('road sen ' + str(sensor) + ' m: ' + 
              'dist ' + str(d+1) + '/' + str(DEV_DIST_GRID) + 
              '; ang ' + str(a+1) + '/' + str(DEV_ANG_GRID) + 
              '; vel ' + str(v+1) + '/' + str(VEL_GRID) + '\n' +
              'min ' + str(grid[d, a, v, 0]) + 
              '; max ' + str(grid[d, a, v, 1]) + '\n' +
              'avg ' + str(grid[d, a, v, 2]) + 
              '; std ' + str(grid[d, a, v, 3]) + '\n'
              )

    plt.xlabel('data points')
    plt.ylabel('wheel angles')
    plt.plot(angles[d][a][v], colour)
    if angles[d][a][v] != []:
        plt.axis([-2, len(angles[d][a][v]) + 1, -0.5, 0.5])
    plt.show()

def std_categories_plot(fig_num):
    plt.figure(fig_num)
    plt.xlabel('categories')
    plt.ylabel('std')
    plt.plot(table_2d_all_0_cc[:, 6], 'blue')
    plt.plot(table_2d_all_5_cc[:, 6], 'red')
    plt.plot(table_2d_all_10_cc[:, 6], 'orange')
    plt.plot(table_2d_all_15_cc[:, 6], 'green')
    plt.axis([-2, len(table_2d_all_10_cc[:, 6]) + 1, -0.05, 0.3])
    plt.show()
    
'''
figure_num = 0
for s in sensors:
    for d in range(DEV_DIST_GRID):
        for a in range(DEV_ANG_GRID):
                plot_group(s, d, a, figure_num)
                figure_num += 1
s = sensors[0]
d = 0
a = 0
plot_group(s, d, a, figure_num)

for s in sensors:
    for d in range(DEV_DIST_GRID):
        for a in range(DEV_ANG_GRID):
            for v in range(VEL_GRID):
                plot_group_v(s, d, a, v, figure_num)
                figure_num += 1
s = sensors[0]
d = 0
a = 0
v = 0
plot_group_v(s, d, a, v, figure_num)
'''
