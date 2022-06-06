import numpy as np

g_l = np.load("glob_losses_np.npy", allow_pickle=True)
good_rows = []
for row in g_l:
    if row[5] < 0.01:
        good_rows.append(row)
np_gl = np.array(good_rows)