#!/usr/bin/env python

"""
Usage:
  merge.py <peak_list_dir> <spind_indexing_dir> <point_group> [options]

Options:
  -h --help                         Show this screen.
  -v --verbose                      Verbose output.
  -o --output=output_file           Output filename [default: merge.hkl].
  --odd-only                        Merge odd events only.
  --even-only                       Merge even events only.
  --pixel-size=pixel_size           Pixel size in meters [default: 110E-6].
  --wave-length=wave_length         Wave length in meters[default: 1.280e-10].
  --det-dist=det_dist               Detector distance in meters [default: 92.40E-3]. 
  --min-match-rate=min_match_rate   Min match rate accepted while merging [default: 0.5].
  --eval-tol=eval_tol               HKL tolerence between observed peaks and predicted spots [default: 0.25].
  --hkl-diag                        Output eHKL for debug.
"""

import numpy as np 
import math
from docopt import docopt
from glob import glob
import os
import sys
from tqdm import tqdm


def get_hkl(q, A=None, A_inv=None):
  """calculate hkl from q vectors
  
  Args:
      q (ndarray, [N, 3]): fourier vectors
      A (ndarray, [3, 3], optional): transformational matrix
      A_inv (ndarray, [3, 3], optional): inverse transformational matrix
  
  Returns:
      ndarray, [N, 3]: hkl
  """
  if A_inv is not None:
    hkl = A_inv.dot(q.T)
  else:
    assert A is not None  # must provide A or A_inv
    A_inv = np.linalg.inv(A)
    hkl = A_inv.dot(q.T)
  return hkl.T


def det2fourier(det_xy, wave_length, det_dist):
  """Detector 2d coordinates to fourier 3d coordinates
  
  Args:
      det_xy (TYPE): Description
      wave_length (TYPE): Description
      det_dist (TYPE): Description
  
  Returns:
      TYPE: 3d fourier coordinates in angstrom^-1
  
  """
  nb_xy = len(det_xy)
  det_dist = np.ones(nb_xy) * det_dist
  det_dist = np.reshape(det_dist, (-1, 1))
  q1 = np.hstack((det_xy, det_dist))
  q1_norm = np.sqrt(np.diag(q1.dot(q1.T)))
  q1_norm = q1_norm.reshape((-1, 1)).repeat(3, axis=1)
  q1 = q1 / q1_norm
  q0 = np.asarray([0., 0., 1.])
  q0 = q0.reshape((1,-1)).repeat(nb_xy, axis=0)
  q = 1. / wave_length * (q1 - q0)
  return q


def hkl2int_(h, k, l):
    int_ = 1000000*int(h) + 1000*int(k) + l 
    return int_


def int_2hkl(int_):
    int_ = int(int_)
    h = int_ // 1000000
    k = (int_%1000000)//1000
    l = int_%1000
    return h,k,l


class Reflection(object):
    """docstring for Reflection"""
    def __init__(self, h, k, l):
        self.h = int(h)
        self.k = int(k)
        self.l = int(l)
        self.int_ = hkl2int_(h, k, l)
        self.intensity_list = []

    def add_measure(self, intensity):
        """add new measurement of this reflection
        
        Args:
            intensity (TYPE): Description
        """
        self.intensity_list.append(intensity)

    def __str__(self):
        return '%d %d %d %s' % (self.h, self.k, self.l, self.intensity_list)


def write2hkl(ref_dict, pointgroup, hkl_file='merge.hkl'):
    import operator
    sorted_refls = sorted(ref_dict.items(), key=operator.itemgetter(0))
    f = open(hkl_file, 'w')
    f.write('CrystFEL reflection list version 2.0\n')
    f.write('Symmetry: %s\n' % point_group)
    f.write('   h    k    l          I    phase   sigma(I)   nmeas\n')
    for i in tqdm(range(len(sorted_refls))):
        refl = sorted_refls[i][1]
        intensity = np.mean(refl.intensity_list)
        redundancy = len(refl.intensity_list)
        int_sigma = np.std(refl.intensity_list) / math.sqrt(redundancy)
        nmeas = len(refl.intensity_list)
        if nmeas > 1:  # remove reflections with only one measurement
          f.write('%4d%5d%5d%11.2f%9s%11.2f%8d\n' %
              (refl.h, refl.k, refl.l, intensity, '-', int_sigma, nmeas))
    f.write('End of reflections\n')
    f.close()
        

if __name__ == '__main__':
    # parse command arguments
    argv = docopt(__doc__)
    spind_dir = argv['<spind_indexing_dir>']  # spind indexing result
    peak_dir = argv['<peak_list_dir>']  # peak lists extracted from cxi
    point_group = argv['<point_group>']
    pixel_size = float(argv['--pixel-size'])
    det_dist = float(argv['--det-dist'])
    wave_length = float(argv['--wave-length'])
    min_match_rate = float(argv['--min-match-rate'])
    eval_tol = float(argv['--eval-tol'])
    output_file = argv['--output']
    verbose = argv['--verbose']
    hkl_diag = argv['--hkl-diag']

    nb_merge = 0  # number of merged pattern
    eHKLs = []
    peak_files = glob('%s/*.txt' % peak_dir)
    reflection_dict = {}

    print('Processing peak files')
    for i in tqdm(range(len(peak_files))):
        if argv['--odd-only']:
            if i % 2 == 0:
                continue
        elif argv['--even-only']:
            if i % 2 == 1:
                continue
        peak_file = peak_files[i]

        peak_data = np.loadtxt(peak_file)
        peak_coord_pixel = peak_data[:,:2]
        peak_coord = peak_coord_pixel * pixel_size
        intensity = peak_data[:,2]
        qs = det2fourier(peak_coord, wave_length, det_dist)

        peak_basename = os.path.basename(peak_file)
        event_id = int(peak_basename[-8:-4])
        spind_file = os.path.join(spind_dir, '%s-spind.txt' %
            peak_basename[:-10])
        if not os.path.exists(spind_file):
            continue
        spind_data = np.loadtxt(spind_file)
        event_ids = spind_data[:,0].astype(np.int)
        row = np.where(event_ids == event_id)[0]
        if row.size == 0:
            if verbose:
                print('event %s not found in %s' % 
                    (event_id, os.path.basename(spind_file)))
            continue
        else:
            assert row.size == 1
            row = row[0]
            match_rate = spind_data[row][1]
            if match_rate < min_match_rate:
                if verbose:
                    print('event %s failed in match rate test in %s' %
                        (event_id, os.path.basename(spind_file)))
                continue
            # pass match rate test
            nb_merge += 1
            A = np.ones((3,3))
            A[:,0] = spind_data[row][4:7]
            A[:,1] = spind_data[row][7:10]
            A[:,2] = spind_data[row][10:13]
            HKL = get_hkl(qs, A=A)  # decimal hkls
            rHKL = np.round(HKL)
            eHKL = np.abs(HKL - rHKL)
            eHKLs += eHKL.tolist()
            pair_ids = np.where(eHKL.max(axis=1) < eval_tol)[0]
            nb_pair = len(pair_ids)
            if float(nb_pair) / float(peak_data.shape[0]) < min_match_rate:
                if verbose:
                    print('Warning: match rate recalculated failed test!!!')
            for j, pair_id in enumerate(pair_ids):
                hkl = rHKL[pair_id].astype(np.int)
                if point_group == 'mmm':
                    h,k,l = abs(hkl[0]), abs(hkl[1]), abs(hkl[2])
                elif point_group == '2':
                    h = abs(hkl[0])
                    k = hkl[1]
                    if hkl[0] >= 0:
                        l = hkl[2]
                    else:
                        l = -hkl[2]
                else:
                    print('point group %s not implemented' % point_group)
                    sys.exit()
                int_ = hkl2int_(h, k, l)
                if reflection_dict.has_key(int_):
                    reflection_dict[int_].add_measure(intensity[pair_id])
                else:
                    reflection = Reflection(h, k, l)
                    reflection.add_measure(intensity[pair_id])
                    reflection_dict[reflection.int_] = reflection

    print('%d events merged.' % nb_merge)

    # write to hkl file
    print('write hkl to %s' % output_file)
    write2hkl(reflection_dict, point_group, output_file)

    # write eHKLs if specified
    if hkl_diag:
        np.savetxt('eHKL.txt', eHKLs, fmt='%.3f %.3f %.3f')
