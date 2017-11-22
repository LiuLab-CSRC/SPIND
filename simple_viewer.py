import numpy as np 
from numpy.linalg import norm
import matplotlib.pyplot as plt 
from math import sqrt, tan, sin, cos, asin
import sys
import itertools


basic_index = sys.argv[1]
event_id = int(sys.argv[2])
detailed_index = sys.argv[3]
crystal_id = int(sys.argv[4])

f1 = np.loadtxt(basic_index)
f2 = np.loadtxt(detailed_index)
hkls = np.rint(f2[np.where(f2[:,-1]==1.)[0],2:5]).astype(np.int).tolist()
hkls.sort()
unique_hkl = list(hkl for hkl,_ in itertools.groupby(hkls))
print('%d unique reflections found' % len(unique_hkl))

crystal_rows = np.where(f1[:,0].astype(np.int) == event_id)[0]
crystal_row = np.where(f1[crystal_rows,1].astype(np.int) == crystal_id)[0][0]
A = np.zeros((3,3))
A[:,0] = f1[crystal_row,5:8]
A[:,1] = f1[crystal_row,8:11]
A[:,2] = f1[crystal_row,11:14]


pixel_size = 100E-6  # 100um pixel
det_dist = 5.109575
lamb = 0.0223904E-10 
peaks_xy = f2[:, :2] / pixel_size

fig = plt.figure('Simple Viewer')
ax = fig.add_subplot(111)
plt.scatter(peaks_xy[:,0], peaks_xy[:,1], marker='o', s=10, label='peaks')
plt.axis('equal')

match_ids = np.where(f2[:,-1] == 1.)[0]
exp_spots = []
pred_spots = []
for i in match_ids:
    peak = peaks_xy[i]
    hkl = f2[i,2:5]
    rhkl = np.rint(hkl)
    q = A.dot(rhkl)
    q_xy = sqrt(q[0] ** 2 + q[1] ** 2)
    q_len = norm(q)
    det_r = det_dist * tan(2. * asin(q_len * lamb * 0.5))
    det_x = det_r * q[0] / q_xy / pixel_size
    det_y = det_r * q[1] / q_xy / pixel_size
    exp_spot = [peak[0], peak[1]]
    pred_spot = [det_x, det_y]
    exp_spots.append(exp_spot)
    pred_spots.append(pred_spot)
    ax.annotate('%.2f %.2f %.2f' % (hkl[0], hkl[1], hkl[2]), 
        xy=(det_x, det_y), xytext=(peak[0], peak[1]))
    ax.annotate('%d %d %d' % (rhkl[0], rhkl[1], rhkl[2]), 
        xy=(det_x, det_y), xytext=(det_x, det_y))

exp_spots = np.array(exp_spots)
pred_spots = np.array(pred_spots)
plt.scatter(exp_spots[:,0], exp_spots[:,1], marker='x', s=50, c='r', label='Exp. spot')
plt.scatter(pred_spots[:,0], pred_spots[:,1], marker='+', s=50, c='g', label='Pred. spot')
plt.legend()
plt.show()
