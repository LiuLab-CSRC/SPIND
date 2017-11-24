import numpy as np 
from numpy.linalg import norm
import matplotlib.pyplot as plt 
from math import sqrt, tan, sin, cos, asin
import sys
import itertools


spind_file = sys.argv[1]
peak_file = sys.argv[2]
event_id = int(sys.argv[3])
crystal_id = int(sys.argv[4])
hkl_tol = float(sys.argv[5])

solutions = np.loadtxt(spind_file)
solution = solutions[np.where((solutions[:,0] == event_id) * (solutions[:,1] == crystal_id))].reshape((14))
peaks = np.loadtxt(peak_file)
qs = peaks[:,4:7]

A  = np.zeros((3,3))
A[:,0] = solution[5:8]
A[:,1] = solution[8:11]
A[:,2] = solution[11:14]

A_inv = np.linalg.inv(A)
hkls = A_inv.dot(qs.T).T
rhkls = np.rint(hkls)
ehkls = np.abs(hkls - rhkls)
pair_ids = np.where(np.max(ehkls, axis=1) < hkl_tol)[0]
print('%d pair peaks found: %s' % (len(pair_ids), str(pair_ids)))

pixel_size = 100E-6  # 100um pixel
det_dist = 5.109575
lamb = 0.0223904E-10 
peaks_xy = peaks[:, :2] / pixel_size

fig = plt.figure('Simple Viewer')
ax = fig.add_subplot(111)
plt.scatter(peaks_xy[:,0], peaks_xy[:,1], marker='o', s=10, label='peaks')
plt.axis('equal')

exp_spots = []
pred_spots = []
for i in pair_ids:
    peak = peaks_xy[i]
    hkl = hkls[i]
    rhkl = np.rint(hkl)
    pred_q = A.dot(rhkl)
    pred_qxy = sqrt(pred_q[0] ** 2 + pred_q[1] ** 2)
    pred_qlen = norm(pred_q)
    pred_detr = det_dist * tan(2. * asin(pred_qlen * lamb * 0.5))
    pred_detx = pred_detr * pred_q[0] / pred_qxy / pixel_size
    pred_dety = pred_detr * pred_q[1] / pred_qxy / pixel_size
    exp_spot = [peak[0], peak[1]]
    pred_spot = [pred_detx, pred_dety]
    exp_spots.append(exp_spot)
    pred_spots.append(pred_spot)
    ax.annotate('%.2f %.2f %.2f' % (hkl[0], hkl[1], hkl[2]), 
        xy=(pred_detx, pred_dety), xytext=(peak[0], peak[1]))
    # ax.annotate('%d %d %d' % (rhkl[0], rhkl[1], rhkl[2]), 
    #     xy=(pred_detx, pred_dety), xytext=(pred_detx, pred_dety))

exp_spots = np.array(exp_spots)
pred_spots = np.array(pred_spots)
plt.scatter(exp_spots[:,0], exp_spots[:,1], marker='x', s=50, c='r', label='Exp. spot')
plt.scatter(pred_spots[:,0], pred_spots[:,1], marker='+', s=50, c='g', label='Pred. spot')
plt.legend()
plt.show()
