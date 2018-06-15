# Copyright © 2018 Liu Lab, Beijing Computational Science Research Center <http://liulab.csrc.ac.cn>

# Authors:
#   2017-2018      Xuanxuan Li <lxx2011011580@gmail.com>
#   2017-2018      Chufeng Li <chufengl@asu.edu>
#   2017      Richard Kirian <rkirian@asu.edu>
#   2017      Nadia Zatsepin <Nadia.Zatsepin@asu.edu>
#   2017      John Spence <spence@asu.edu>
#   2017      Haiguang Liu <hgliu@csrc.ac.cn>

# This file is part of SPIND.

# SPIND is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPIND is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with SPIND.  If not, see <http://www.gnu.org/licenses/>.


"""
Usage:
  merge.py <peak_list_dir> <spind_indexing_dir> <point_group> <det_dist>[options]

Options:
  -h --help                         Show this screen.
  -v --verbose                      Verbose output.
  -o --output=output_file           Output filename [default: merge.hkl].
  --odd-only                        Merge odd events only.
  --even-only                       Merge even events only.
  --pixel-size=pixel_size           Pixel size in meters [default: 110E-6].
  --min-match-rate=min_match_rate   Min match rate accepted while merging [default: 0.5].
  --eval-tol=eval_tol               HKL tolerence between observed peaks and predicted spots [default: 0.25].
  --diag                            Output more information for diagnosis.
"""

import numpy as np 
import h5py
import math
from docopt import docopt
from glob import glob
import os
import sys
from tqdm import tqdm


h_ = 4.135667662E-15  # Planck constant in eV*s
c_ = 2.99792458E8  # light speed in m/sec


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
    for i in range(len(sorted_refls)):
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
    det_dist = float(argv['<det_dist>'])
    # wave_length = float(argv['<wave_length>'])
    min_match_rate = float(argv['--min-match-rate'])
    eval_tol = float(argv['--eval-tol'])
    output_file = argv['--output']
    verbose = argv['--verbose']
    diag = argv['--diag']

    nb_merge = 0  # number of merged pattern
    nb_event = 0  # number of event processed
    nb_meas = 0  # number of measurements merged
    eHKLs = []  # HKL error
    RESs = []  # resolution of reflections in accepted events
    peak_files = glob('%s/*.h5' % peak_dir)
    reflection_dict = {}
    spind_dict = {}

    # loading all the SPIND indexing files
    spind_files = glob('%s/*spind.txt' % spind_dir)
    for spind_file in spind_files:
        basename = os.path.basename(spind_file)
        spind_dict[basename[:-10]] = np.loadtxt(spind_file)

    for i in range(len(peak_files)):
        peak_file = peak_files[i]
        print('Processing %s' % peak_file)
        peak_h5 = h5py.File(peak_file, 'r')
        for key in peak_h5.keys():
            peak_data = peak_h5[key].value
            peak_coord_pixel = peak_data[:,:2]
            peak_coord = peak_coord_pixel * pixel_size
            intensity = peak_data[:,2]
            photon_energy = peak_data[0,4]  # in eV
            wave_length = h_ * c_ / photon_energy
            qs = det2fourier(peak_coord, wave_length, det_dist) 

            event_id = int(key[-4:])
            spind_key = key[:-6]
            if not spind_dict.has_key(spind_key):
                continue
            spind_data = spind_dict[spind_key]
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
                nb_event += 1
                if argv['--odd-only'] and (nb_event % 2) == 0:
                    continue
                elif argv['--even-only'] and (nb_event % 2) == 1:
                    continue
                nb_merge += 1
                A = np.ones((3,3))
                A[:,0] = spind_data[row][4:7]
                A[:,1] = spind_data[row][7:10]
                A[:,2] = spind_data[row][10:13]
                HKL = get_hkl(qs, A=A)  # decimal hkls
                rHKL = np.round(HKL)
                eHKL = np.abs(HKL - rHKL)
                eHKLs += eHKL.tolist()
                res = 1. / np.sqrt(np.diag(qs.dot(qs.T))) * 1E10  # in Angstrom
                RESs += res.tolist()
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
                    nb_meas += 1
                    if reflection_dict.has_key(int_):
                        reflection_dict[int_].add_measure(intensity[pair_id])
                    else:
                        reflection = Reflection(h, k, l)
                        reflection.add_measure(intensity[pair_id])
                        reflection_dict[reflection.int_] = reflection

    print('%d events processed, %d event merged with %d measurements.' % 
        (nb_event, nb_merge, nb_meas))

    # write to hkl file
    print('write hkl to %s' % output_file)
    write2hkl(reflection_dict, point_group, output_file)

    # write eHKLs and RESs for diagnosis
    if diag:
        eHKLs = np.array(eHKLs)
        RESs = np.array(RESs).reshape((-1, 1))
        data = np.concatenate((eHKLs, RESs), axis=1)
        np.savetxt('diag.txt', data, fmt='%.3f %.3f %.3f %.3f')
