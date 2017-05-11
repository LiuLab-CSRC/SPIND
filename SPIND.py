#!/bin/env python
"""
Usage:
  SPIND.py <config.yml> -i peak_list_dir -t table_file  [options]

Options:
  -h --help                       Show this screen.
  -i peak_list_dir                Peak list directory.
  -t table_file                   Table filepath containing spot vector length 
                                  and pair angles.
  -o output_dir                   Output directory [default: output].
  --start=<start_event_id>        The first event id to index [default: 0].
  --end=<end_event_id>            The last event id to index [default: last].
  --refine-cycle=<refine_cycle>   Number of refine cycle [default: 10].
  --pair-tol=<pair_tol>           Reciprocal vector length and angle tolerence 
                                  in pair matching [default: 3.E7,1.0].
  --eval-tol=<eval_tol>           hkl tolerence between observed peaks and 
                                  predicted spots [default: 0.25].
"""

from docopt import docopt
import yaml
import logging
from mpi4py import MPI
import os
import glob
import numpy as np
import h5py
from math import acos, pi, cos, sin
from numpy.linalg import norm
from scipy.optimize import fmin_cg
from itertools import combinations


def parse_peak_list_filename(filename):
  basename = os.path.splitext(os.path.basename(filename))[0]
  experiment, run_id, class_id, event_id = basename.split('-')
  return experiment, run_id, class_id


def load_peaks(filepath):
  peaks = np.loadtxt(filepath)
  return peaks


def load_table(filepath):
  print("loading table: %s" % filepath)
  table = h5py.File(filepath, 'r')
  table_dict = {}
  table_dict['hkl1'] = table['hkl1'].value.astype(np.int16)
  table_dict['hkl2'] = table['hkl2'].value.astype(np.int16)
  table_dict['LA'] = table['LA'].value.astype(np.float32)
  return table_dict


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


def rad2deg(rad):
  return float(rad) / pi * 180.


def deg2rad(deg):
  return float(deg) / 180. * pi


def calc_transform_matrix(cell_parameters):
  a, b, c = np.asarray(cell_parameters[0:3])
  al, be, ga = cell_parameters[3:]
  # monoclinic
  av = [a, 0., 0.]
  bv = [0, b, 0.]
  cv = [c*cos(deg2rad(be)), 0, c*sin(deg2rad(be))]
  a_star = (np.cross(bv, cv)) / ((np.cross(bv, cv).dot(av)))
  b_star = (np.cross(cv, av)) / ((np.cross(cv, av).dot(bv)))
  c_star = (np.cross(av, bv)) / ((np.cross(av, bv).dot(cv)))
  A = np.zeros((3, 3), dtype=np.float64)  # transform matrix
  A[:,0] = a_star
  A[:,1] = b_star
  A[:,2] = c_star
  return A


def calc_rotation_matrix(q1, q2, ref_q1, ref_q2):
  """Calculate rotation matrix R so that R.dot(ref_q1,2) ~= q1,2
  
  argv:
    q1 (TYPE): Description
    q2 (TYPE): Description
    ref_q1 (TYPE): Description
    ref_q2 (TYPE): Description
  
  Returns:
    TYPE: Description
  """
  ref_nv = np.cross(ref_q1, ref_q2) 
  q_nv = np.cross(q1, q2)
  if min(norm(ref_nv), norm(q_nv)) == 0.:  # avoid 0 degree including angle
    return np.identity(3)
  axis = np.cross(ref_nv, q_nv)
  angle = rad2deg(acos(ref_nv.dot(q_nv) / (norm(ref_nv) * norm(q_nv))))
  R1 = axis_angle_to_rotation_matrix(axis, angle)
  rot_ref_q1, rot_ref_q2 = R1.dot(ref_q1), R1.dot(ref_q2)  # rotate ref_q1,2 plane to q1,2 plane

  cos1 = max(min(q1.dot(rot_ref_q1) / (norm(rot_ref_q1) * norm(q1)), 1.), -1.)  # avoid math domain error
  cos2 = max(min(q2.dot(rot_ref_q2) / (norm(rot_ref_q2) * norm(q2)), 1.), -1.)
  angle1 = rad2deg(acos(cos1))
  angle2 = rad2deg(acos(cos2))
  angle = (angle1 + angle2) / 2.
  axis = np.cross(rot_ref_q1, q1)
  R2 = axis_angle_to_rotation_matrix(axis, angle)

  R = R2.dot(R1)
  return R


def axis_angle_to_rotation_matrix(axis, angle):
  """Convert axis angle to rotation matrix
  
  argv:
    axis (TYPE): Rotation axis vector
    angle (TYPE): Rotation angle in degrees
  
  Returns:
    TYPE: Rotation matrix
  """
  x, y, z = axis / norm(axis)
  angle = deg2rad(angle)
  c, s = cos(angle), sin(angle)
  R = [[c+x**2.*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
     [y*x*(1-c)+z*s, c+y**2.*(1-c), y*z*(1-c)-x*s],
     [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z**2.*(1-c)]]
  return np.asarray(R)


def eval_solution(R, qs, A0_inv, eval_tol=0.25):
  """Calculate match rate for the given rotation matrix
  
  argv:
    R (TYPE): Rotation matrix
    qs (TYPE): Peak coorinates in fourier space
    A0_inv (TYPE): Description
    eval_tol (float, optional): hkl tolerence
  
  Returns:
    TYPE: Description
  """
  R_inv = np.linalg.inv(R)
  hkls = A0_inv.dot(R_inv.dot(qs.T)).T
  rhkls = np.rint(hkls)
  ehkls = np.abs(hkls - rhkls)
  pair_ids = np.where(np.max(ehkls, axis=1) < eval_tol)[0]  # indices of matched peaks
  nb_pairs = len(pair_ids)
  nb_peaks = len(qs)
  match_rate = float(nb_pairs) / float(nb_peaks)
  return match_rate, nb_pairs, hkls, rhkls, pair_ids


def calc_angle(v1, v2):
  l1, l2 = norm(v1), norm(v2)
  if min(l1, l2) < 1E-15:  # 0 vector
    return 0.
  cos_value = v1.dot(v2) / (l1 * l2)
  cos_value = max(min(cos_value, 1.), -1.)
  angle = rad2deg(acos(cos_value))
  return angle


def index(peaks, table, A0, A0_inv, 
          pair_tol=(3.E7, 1.0), eval_tol=0.25, refine_cycle=10):
  """Summary
  
  argv:
    peaks (TYPE): Description
    table (numpy.ndarray): Description
    A (TYPE): transform_matrix
    A_inv (TYPE): Description
    pair_tol (list, optional): Description
    eval_tol (float, optional): Description
    refine_cycle (int, optional): Description
  
  Returns:
    TYPE: Description
  """
  match_rate = 0.
  nb_pairs = 0  # number of matched pairs
  hkls = None  # decimal Miller indices
  rhkls = None   # interger Miller indices
  pair_ids = None  # indices of matched peaks
  R = np.identity(3)
  peaks_xy = peaks[:,0:2] * pixel_size
  qs = det2fourier(peaks_xy, wave_length, detector_distance)

  pair_pool = list(combinations(range(5), 2))
  print(table['LA'][:10])
  for i in range(len(pair_pool)):
    pair = pair_pool[i]
    q1, q2 = qs[pair[0]], qs[pair[1]]
    q1_norm, q2_norm = norm(q1), norm(q2)
    if q1_norm < q2_norm:
        q1, q2 = q2, q1
        q1_norm, q2_norm = q2_norm, q1_norm
    angle = calc_angle(q1, q2)
    match_ids = np.where((np.abs(q1_norm - table['LA'][:,0]) < pair_tol[0]) * 
               (np.abs(q2_norm - table['LA'][:,1]) < pair_tol[0]) *
               (np.abs(angle - table['LA'][:,2]) < pair_tol[1]))[0]
    for match_id in match_ids:
      hkl1 = table['hkl1'][match_id]
      hkl2 = table['hkl2'][match_id]
      ref_q1, ref_q2 = A0.dot(hkl1), A0.dot(hkl2)
      _R = calc_rotation_matrix(q1, q2, ref_q1, ref_q2)
      _match_rate, _nb_pairs, _hkls, _rhkls, _pair_ids = eval_solution(
        _R, qs, A0_inv, eval_tol=eval_tol)
      if _match_rate > match_rate:
        match_rate, nb_pairs = _match_rate, _nb_pairs
        hkls, rhkls, pair_ids = _hkls, _rhkls, _pair_ids
        R = _R
  if hkls is None:
    eXYZs = np.ones((len(qs), 3), dtype=np.float32)
  else:
    eXYZs = np.abs(A0.dot(hkls.T) - A0.dot(rhkls.T)).T  # Fourier space error between peaks and predicted spots
  dists = np.sqrt(eXYZs.dot(eXYZs.T).diagonal())
  pair_dist = np.mean(dists[pair_ids])  # average distance between mached peaks and the correspoding predicted spots
  A = R.dot(A0)
  if pair_ids is None:
    pair_dist_refined, A_refined = pair_dist, A
  else:
    pair_dist_refined, A_refined = refine(A, rhkls, qs, pair_ids, refine_cycle)  # refine A matrix with matched pairs to minimize norm(AH-q)
  logging.info("After refinement, delta_A %.3e, dist %.3e -> %.3e" % 
         (norm(A_refined - A), pair_dist, pair_dist_refined))
  return A, match_rate, nb_pairs, pair_dist, A_refined, pair_dist_refined


def refine(A, hkls, qs, pair_ids, refine_cycle):
  A_refined = A.copy()
  def _fun(x, *argv):
    asx, bsx, csx, asy, bsy, csy, asz, bsz, csz = x
    h, k, l, qx, qy, qz = argv
    r1 = (asx*h + bsx*k + csx*l - qx)
    r2 = (asy*h + bsy*k + csy*l - qy)
    r3 = (asz*h + bsz*k + csz*l - qz)
    return r1**2. + r2**2. + r3**2.

  def _gradient(x, *argv):
    asx, bsx, csx, asy, bsy, csy, asz, bsz, csz = x
    h, k, l, qx, qy, qz = argv
    r1 = (asx*h + bsx*k + csx*l - qx)
    r2 = (asy*h + bsy*k + csy*l - qy)
    r3 = (asz*h + bsz*k + csz*l - qz)
    g_asx, g_bsx, g_csx = 2.*h*r1, 2.*k*r1, 2.*l*r1
    g_asy, g_bsy, g_csy = 2.*h*r2, 2.*k*r2, 2.*l*r2
    g_asz, g_bsz, g_csz = 2.*h*r3, 2.*k*r3, 2.*l*r3
    return np.asarray((g_asx, g_bsx, g_csx,
               g_asy, g_bsy, g_csy,
               g_asz, g_bsz, g_csz))
  for i in range(refine_cycle):
    for j in range(len(pair_ids)):  # refine by each reflection
      pair_id = pair_ids[j]
      x0 = A_refined.reshape((-1))
      hkl = hkls[pair_id,:]
      q = qs[pair_id,:]
      args = (hkl[0], hkl[1], hkl[2], q[0], q[1], q[2])
      res = fmin_cg(_fun, x0, fprime=_gradient, args=args, disp=0)
      A_refined = res.reshape((3,3))
    eXYZs = np.abs(A_refined.dot(hkls.T) - qs.T).T
    dists = np.sqrt(eXYZs.dot(eXYZs.T).diagonal())
    pair_dist = np.mean(dists[pair_ids])
  return pair_dist, A_refined


if __name__ == '__main__':
  # parse command options
  argv = docopt(__doc__)
  config_file = argv['<config.yml>']
  peak_list_dir = argv['-i']
  table_filepath = argv['-t']
  output_dir = argv['-o']
  refine_cycle = int(argv['--refine-cycle'])
  start_id = int(argv['--start'])
  end_id = argv['--end']
  if end_id == 'last':
    end_id = len(glob.glob(peak_list_dir + '/*.txt')) - 1
  else:
    end_id = int(end_id)
  pair_tol_str = argv['--pair-tol']
  pair_tol_list = pair_tol_str.split(',')
  pair_tol = np.asarray(pair_tol_list, dtype=np.float)
  eval_tol = float(argv['--eval-tol'])

  # load configurations
  config = yaml.load(open(config_file, 'r'))  
  wave_length = config['wave length']
  detector_distance = config['detector distance']
  pixel_size = config['pixel size'] 
  cell_parameters = np.asarray(config['cell parameters']) 
  cell_parameters[:3] *= 1E-10  # convert to meters
  A0 = calc_transform_matrix(cell_parameters)
  A0_inv = np.linalg.inv(A0)

  experiment, run_id, class_id = parse_peak_list_filename(
    glob.glob(peak_list_dir + '/*.txt')[0])
  prefix = experiment + '-' + run_id + '-' + class_id
  
  # MPI setup
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  # master sends jobs
  if rank == 0:
    # mkdir for output
    if os.path.isdir(output_dir):
      pass
    else:
      try:
        os.makedirs('%s' %output_dir)
      except Exception as e:
        raise e
    logging.basicConfig(filename=os.path.join(
        output_dir, 'debug-%d.log' % rank),level=logging.DEBUG)
    logging.info(str(argv))
    # assign jobs to slaves and master itself
    nb_patterns = end_id - start_id + 1
    job_size = nb_patterns // size 
    jobs = []
    for i in range(size):
      if i == (size - 1):
        job = np.arange(i*job_size+start_id, end_id+1)
      else:
        job = np.arange(i*job_size+start_id, (i+1)*job_size+start_id)
      jobs.append(job)
      if i == 0:
        continue
      else:
        comm.send(job, dest=i)
        logging.info('Rank 0 send job to rank %d: %s' % (i, str(job)))
    job = jobs[0] 
  # slaves receive jobs
  else:
    job = comm.recv(source=0)
    logging.basicConfig(filename=os.path.join(
        output_dir, 'debug-%d.log' % rank),level=logging.DEBUG)
    logging.info('Rank %d receive job: %s' % (rank, str(job)))

  # workers do assigned jobs
  logging.info('Rank %d processing jobs: %s' % (rank, str(job)))
  table = load_table(table_filepath)
  nb_indexed = 0
  indexed_events = []
  output = open(os.path.join(output_dir, 
    'spind_indexing-%d.txt' % rank), 'w')
  for i in range(len(job)):
    event_id = job[i]
    filename = prefix + '-e%04d' % event_id + '.txt'
    filepath = os.path.join(peak_list_dir, filename)
    if not os.path.exists(filepath):
      logging.warning('peak list file %s do not exist' % filepath)
    else:
      logging.info('Rank %d working on event %04d: %s' % 
        (rank, event_id, filepath))
      peaks = load_peaks(filepath)
      A, match_rate, nb_pairs, pair_dist, A_refined, pair_dist_refined = index(
        peaks, table, A0, A0_inv, pair_tol=pair_tol, eval_tol=eval_tol, 
        refine_cycle=refine_cycle
      )
      logging.info('Event %04d, match rate %.2f' % (event_id, match_rate))
      logging.info('A: %s' % str(A))
      if pair_dist_refined >= pair_dist:
        best_A = A 
        best_dist = pair_dist
      else:
        best_A = A_refined
        best_dist = pair_dist_refined
      output.write('%6d %.2f %4d %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E\n'
             % (job[i], match_rate, nb_pairs, best_dist,
              best_A[0,0], best_A[1,0], best_A[2,0],
              best_A[0,1], best_A[1,1], best_A[2,1],
              best_A[0,2], best_A[1,2], best_A[2,2],
              pair_dist))
      if match_rate >= 0.5:
        nb_indexed += 1
        indexed_events.append(event_id)
      print('Event %d nb_peak %d, match rate %.2f, pair dist: %.3e -> %.3e' 
          % (job[i], len(peaks), match_rate, pair_dist, pair_dist_refined))
    logging.info('Rank %d indexing rate: %.2f%%' % (rank, nb_indexed*100/(i+1)))
  print('Rank %d has %d indexed: %s with match score higher than 50%%' % 
    (rank, nb_indexed, str(indexed_events)))
  output.close()

  comm.barrier()
  # merge indexing results to single txt file
  if rank == 0:
    data = np.array([])
    for i in range(size):
      if i == 0:
        data = np.loadtxt(os.path.join(output_dir, 'spind_indexing-0.txt'))
      else:
        data = np.concatenate((data, np.loadtxt(os.path.join(output_dir, 
            'spind_indexing-%d.txt' % i))), axis=0)
    np.savetxt(os.path.join(output_dir, prefix + '-spind.txt'),
           data[:,0:-1], fmt="%6d %.2f %4d %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E")
    overall_indexing_rate = float((data[:,1] > 0.5).sum()) / float(data.shape[0]) * 100.
    print('Overall indexing rate: %.2f%%' % overall_indexing_rate)