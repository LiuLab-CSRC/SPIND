#!/usr/bin/env python
#! coding=utf-8

"""
Usage: 
  generate_table.py <config.yml> [options]

Options:
  -h --help             Show this screen.
  -o table_file         Output table [default: spind_table.h5].
"""

import numpy as np 
from math import acos, pi, cos, sin
from numpy.linalg import norm
import h5py
import itertools

from docopt import docopt
import yaml
from tqdm import tqdm

from exceptions import NotImplementedError


def rad2deg(rad):
  return float(rad) / pi * 180.


def deg2rad(deg):
  return float(deg) / 180. * pi


def calc_transform_matrix(cell_param, 
                          lattice_type=None):
  a, b, c = np.asarray(cell_param[0:3])
  al, be, ga = cell_param[3:]
  if lattice_type == 'monoclinic':
    av = [a, 0., 0.]
    bv = [0, b, 0.]
    cv = [c*cos(deg2rad(be)), 0, c*sin(deg2rad(be))]
  elif lattice_type == 'orthorhombic':
    av = [a, 0., 0.]
    bv = [0., b, 0.]
    cv = [0., 0., c]
    assert al == 90.
    assert be == 90.
    assert ga == 90.
  else:
    raise NotImplementedError('%s not implemented yet' %
      lattice_type)
  a_star = (np.cross(bv, cv)) / ((np.cross(bv, cv).dot(av)))
  b_star = (np.cross(cv, av)) / ((np.cross(cv, av).dot(bv)))
  c_star = (np.cross(av, bv)) / ((np.cross(av, bv).dot(cv)))
  A = np.zeros((3, 3), dtype=np.float64)  # transform matrix
  A[:,0] = a_star
  A[:,1] = b_star
  A[:,2] = c_star
  return A


def calc_angle(v1, v2):
  l1, l2 = norm(v1), norm(v2)
  if min(l1, l2) < 1E-15:  # 0 vector
    return 0.
  cos_value = v1.dot(v2) / (l1 * l2)
  cos_value = max(min(cos_value, 1.), -1.)
  angle = rad2deg(acos(cos_value))
  return angle


if __name__ == '__main__':
  # parse command options
  argv = docopt(__doc__)
  config_file = argv['<config.yml>']
  table_file = argv['-o']
  print('Loading configuration from =%s=' %
    config_file)
  print('Reference table saved to =%s=' %
    table_file)

  # load configurations
  config = yaml.load(open(config_file, 'r'))  
  res_cutoff = config['resolution cutoff']
  lattice_type = config['lattice type']
  cell_param = np.asarray(config['cell parameters'])
  cell_param[:3] *= 1E-10  # convert to meters
  centering = config['centering']
  wave_len = config['wave length']
  det_dist = config['detector distance']
  pixel_size = config['pixel size'] 
  if config.has_key('hkl file'):
    hkl_file = config['hkl file']
    print('Generating reference table from hkl file: %s' %
      hkl_file)
  else:
    hkl_file = None

  # a, b, c star
  A = calc_transform_matrix(cell_param, 
    lattice_type=lattice_type)
  a_star, b_star, c_star = A[:,0], A[:,1], A[:,2] 

  if hkl_file is None:
    # hkl grid
    q_cutoff = 1. / res_cutoff
    max_h = int(np.ceil(q_cutoff / norm(a_star)))
    max_k = int(np.ceil(q_cutoff / norm(b_star)))
    max_l = int(np.ceil(q_cutoff / norm(c_star)))
    print('max_h: %d, max_k: %d, max_l: %d' %
      (max_h, max_k, max_l))    

    hh = np.arange(-max_h, max_h+1)
    kk = np.arange(-max_k, max_k+1)
    ll = np.arange(-max_l, max_l+1)   

    hs, ks, ls = np.meshgrid(hh, kk, ll)
    hkls = np.ones((hs.size, 3))
    hkls[:,0] = hs.reshape((-1))
    hkls[:,1] = ks.reshape((-1))
    hkls[:,2] = ls.reshape((-1))    

    # remove high resolution hkls
    qs = A.dot(hkls.T).T
    valid_idx = []
    for i in range(len(qs)):
      if norm(qs[i]) <= q_cutoff:
        valid_idx.append(i)
    hkls = hkls[valid_idx]    

    # apply systematic absence
    if centering == 'I':  # h+k+l == 2n
      valid_idx = (hkls.sum(axis=1) % 2 == 0)
    elif centering == 'C':  # h+k == 2n
      valid_idx = ((hkls[:,0] + hkls[:,1]) % 2 == 0)
    elif centering == 'A':  # h+k == 2n
      valid_idx = ((hkls[:,1] + hkls[:,2]) % 2 == 0)
    elif centering == 'B':  # h+k == 2n
      valid_idx = ((hkls[:,0] + hkls[:,2]) % 2 == 0)
    elif centering == 'P':
      valid_idx = np.ones(hkls.shape[0]) > 0  # all true
    else:
      raise NotImplementedError('%s not implemented' %
        centering)
    hkls = hkls[valid_idx]  
  else:  # load 1/8 hkls from file
    hkl = np.loadtxt(hkl_file, dtype=np.int16)
    hk_l = hkl.copy()
    hk_l[:,2] = -hkl[:,2]  # h,k,-l
    h_kl = hkl.copy()
    h_kl[:,1] = -hkl[:,1]  # h,-k,l
    _hkl = hkl.copy()
    _hkl[:,0] = -hkl[:,0]  # -h,k,l
    h_k_l = -_hkl.copy()  # h,-k,-l
    _h_kl = -hk_l.copy()  # -h,-k,l
    _hk_l = -h_kl.copy()  # -h,k,-l
    _h_k_l = -hkl.copy()  # -h,-k,-l
    hkls = np.vstack((hkl, hk_l, h_kl, _hkl,
                      h_k_l, _h_kl, _hk_l, _h_k_l))
    hkls = hkls.tolist()
    hkls.sort()
    hkls = np.asarray(list(hkl for hkl,_ in itertools.groupby(hkls)))

  # generate table
  table = h5py.File(table_file) 

  qs = A.dot(hkls.T).T
  lens = []
  for i in range(len(qs)):
    lens.append(norm(qs[i]))  

  batch_size = 100000  # save rows in batch mode
  count = 0
  batch_rows = [] 
  for i in tqdm(range(len(hkls))):
    hkl1 = np.int_(hkls[i])
    q1 = qs[i]
    len1 = lens[i]
    for j in range(i+1, len(hkls)):
      hkl2 = np.int_(hkls[j])
      q2 = qs[j]
      len2 = lens[j]
      angle = calc_angle(q1, q2)
      if len1 >= len2:
        row = [hkl1[0], hkl1[1], hkl1[2], 
               hkl2[0], hkl2[1], hkl2[2],
               len1, len2, angle]
      else:
        row = [hkl2[0], hkl2[1], hkl2[2], 
               hkl1[0], hkl1[1], hkl1[2],
               len2, len1, angle]
      batch_rows.append(row)
      count += 1
      if count % batch_size == 0:
        batch_rows = np.asarray(batch_rows)
        if count // batch_size == 1:
          table.create_dataset('hkl1', 
            data=batch_rows[:,0:3].astype(np.int8), maxshape=(None, 3))
          table.create_dataset('hkl2', 
            data=batch_rows[:,3:6].astype(np.int8), maxshape=(None, 3))
          table.create_dataset('LA',   # length and angle
            data=batch_rows[:,6:9].astype(np.float64), maxshape=[None, 3])
        else:
          n = table['hkl1'].shape[0]
          table['hkl1'].resize(n+batch_size, axis=0)
          table['hkl1'][n:count] = np.int_(batch_rows[:,0:3])
          table['hkl2'].resize(n+batch_size, axis=0)
          table['hkl2'][n:count] = np.int_(batch_rows[:,3:6])
          table['LA'].resize(n+batch_size, axis=0)
          table['LA'][n:count] = batch_rows[:,6:9]
        batch_rows = []
  # last batch
  batch_rows = np.asarray(batch_rows)
  n = table['hkl1'].shape[0]
  last_batch_size = count % batch_size
  table['hkl1'].resize(n+last_batch_size, axis=0)
  table['hkl1'][n:count] = batch_rows[:,0:3].astype(np.int8)
  table['hkl2'].resize(n+last_batch_size, axis=0)
  table['hkl2'][n:count] = batch_rows[:,3:6].astype(np.int8)
  table['LA'].resize(n+last_batch_size, axis=0)
  table['LA'][n:count] = batch_rows[:,6:9].astype(np.float64)
  table.close()