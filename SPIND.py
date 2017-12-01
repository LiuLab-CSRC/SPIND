import yaml
import logging
from mpi4py import MPI
import os
import sys
import glob
try:
  import mkl
  mkl.set_num_threads(1)
except:
  pass
import numpy as np
import h5py
from math import acos, pi, cos, sin
from numpy.linalg import norm
from scipy.optimize import fmin_cg
from itertools import combinations
from tqdm import tqdm


h_ = 4.135667662E-15  # Planck constant in eV*s
c_ = 2.99792458E8  # light speed in m/sec
epsilon = 1E-99  


def load_peaks(filepath, sort_by, res_cutoff):
  peaks = np.loadtxt(filepath)
  if sort_by == 'snr':
    ind = np.argsort(peaks[:,3])
  elif sort_by == 'intensity':
    ind = np.argsort(peaks[:,2])
  else:
    print('Please use "intensity" or "snr" sorting method!')
    sys.exit()
  peaks = peaks[ind[::-1]]  # reverse sort
  HP_ind = peaks[:,7] > res_cutoff
  LP_ind = peaks[:,7] <= res_cutoff
  peaks = np.concatenate((peaks[HP_ind], peaks[LP_ind]))
  return peaks


def load_table(filepath):
  table = h5py.File(filepath, 'r')
  table_dict = {}
  table_dict['hkl1'] = table['hkl1'].value.astype(np.int16)
  table_dict['hkl2'] = table['hkl2'].value.astype(np.int16)
  table_dict['LA'] = table['LA'].value.astype(np.float32)
  return table_dict


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


def calc_rotation_matrix(q1, q2, ref_q1, ref_q2):
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
  x, y, z = axis / norm(axis)
  angle = deg2rad(angle)
  c, s = cos(angle), sin(angle)
  R = [[c+x**2.*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
     [y*x*(1-c)+z*s, c+y**2.*(1-c), y*z*(1-c)-x*s],
     [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z**2.*(1-c)]]
  return np.asarray(R)


class Solution(object):
  def __str__(self):
    return str(self.__dict__)


def eval_solution(solution, qs, A0_inv, 
                  eval_tol=0.25,
                  centering=None,
                  centering_factor=0.0,
                  miller_set=None,
                  seed=None,
                  seed_hkl_tol=0.1,
                  indexed_peak_ids=[]):
  R = solution.R
  R_inv = np.linalg.inv(R)
  hkls = A0_inv.dot(R_inv.dot(qs.T)).T
  rhkls = np.rint(hkls)
  ehkls = np.abs(hkls - rhkls)
  solution.hkls = hkls 
  solution.rhkls = rhkls
  solution.ehkls = ehkls
  if miller_set is None:
    pair_ids = np.where(np.max(ehkls, axis=1) < eval_tol)[0]  # indices of matched peaks
  else:
    _pair_ids = np.where(np.max(ehkls, axis=1) < eval_tol)[0]
    pair_ids = []
    for _pair_id in _pair_ids:
      abs_hkl = np.abs(rhkls[_pair_id])
      if norm(miller_set - abs_hkl, axis=1).min() < epsilon:
        pair_ids.append(_pair_id)
  pair_ids = list(set(pair_ids) - set(indexed_peak_ids))

  nb_pairs = len(pair_ids)
  nb_peaks = len(qs)
  match_rate = float(nb_pairs) / float(nb_peaks)
  solution.pair_ids = pair_ids
  solution.match_rate = match_rate
  solution.nb_pairs = nb_pairs
  # centering score
  if nb_pairs == 0:
    centering_score = 0.
  elif centering == 'A':  # k+l=2n
    pair_hkls = rhkls.astype(np.int16)[pair_ids]
    nb_A_peaks = ((pair_hkls[:,1] + pair_hkls[:,2]) % 2 == 0).sum()
    A_ratio = float(nb_A_peaks) / float(nb_pairs)
    centering_score = 2 * A_ratio - 1.
  elif centering == 'B':  # h+l=2n
    pair_hkls = rhkls.astype(np.int16)[pair_ids]
    nb_B_peaks = ((pair_hkls[:,0] + pair_hkls[:,2]) % 2 == 0).sum()
    B_ratio = float(nb_A_peaks) / float(nb_pairs)
    centering_score = 2 * A_ratio - 1.
  elif centering == 'C':  # h+k=2n
    pair_hkls = rhkls.astype(np.int16)[pair_ids]
    nb_C_peaks = ((pair_hkls[:,0] + pair_hkls[:,1]) % 2 == 0).sum()
    C_ratio = float(nb_C_peaks) / float(nb_pairs)
    centering_score = 2 * C_ratio - 1.
  elif centering == 'I':  # h+k+l=2n
    pair_hkls = rhkls.astype(np.int16)[pair_ids]
    nb_I_peaks = (hkls.sum(axis=1) % 2 == 0).sum()
    I_ratio = float(nb_I_peaks) / float(nb_pairs)
    centering_score = 2 * C_ratio - 1.
  elif centering == 'P':
    pair_hkls = rhkls.astype(np.int16)[pair_ids]
    nb_A_peaks = ((pair_hkls[:,1] + pair_hkls[:,2]) % 2 == 0).sum()
    nb_B_peaks = ((pair_hkls[:,0] + pair_hkls[:,2]) % 2 == 0).sum()
    nb_C_peaks = ((pair_hkls[:,0] + pair_hkls[:,1]) % 2 == 0).sum()
    nb_I_peaks = (hkls.sum(axis=1) % 2 == 0).sum()
    centering_score = 2.*(1-float(max(nb_A_peaks, nb_B_peaks, nb_C_peaks, nb_I_peaks))/float(nb_pairs))
  else:
    centering_score = 0.
  solution.centering_score = centering_score

  # evaluation metrics
  solution.seed_error = ehkls[seed,:].max()
  solution.total_score = centering_factor * centering_score + match_rate
  if len(pair_ids) == 0:
    solution.total_error = 1.  # no matching peaks, set error to 1
  else:
    solution.total_error = ehkls[pair_ids].mean()  # naive error of matching peaks

  return solution


def calc_angle(v1, v2):
  l1, l2 = norm(v1), norm(v2)
  if min(l1, l2) < 1E-15:  # 0 vector
    return 0.
  cos_value = v1.dot(v2) / (l1 * l2)
  cos_value = max(min(cos_value, 1.), -1.)
  angle = rad2deg(acos(cos_value))
  return angle
    

def index(peaks, table, A0, A0_inv, 
          seed_pool_size=5,
          seed_len_tol=3.E7, seed_angle_tol=1.0,
          seed_hkl_tol=0.1, eval_tol=0.25,
          centering="P", centering_factor=0., 
          refine_cycles=10,
          miller_set=None, multi_index=False):
  solutions = []
  indexed_peak_ids = []

  if multi_index is True:
    while True:
      solution = index_once(peaks, table, A0, A0_inv, 
                            seed_pool_size=seed_pool_size,
                            seed_len_tol=seed_len_tol, seed_angle_tol=seed_angle_tol,
                            seed_hkl_tol=seed_hkl_tol, eval_tol=eval_tol,
                            centering=centering, centering_factor=centering_factor, 
                            refine_cycles=refine_cycles,
                            miller_set=miller_set, multi_index=multi_index,
                            indexed_peak_ids=indexed_peak_ids)
      if solution.match_rate == 0:
        break
      solutions.append(solution)
      indexed_peak_ids += solution.pair_ids
  else:
    solution = index_once(peaks, table, A0, A0_inv, 
                          seed_pool_size=seed_pool_size,
                          seed_len_tol=seed_len_tol, seed_angle_tol=seed_angle_tol,
                          seed_hkl_tol=seed_hkl_tol, eval_tol=eval_tol,
                          centering=centering, centering_factor=centering_factor, 
                          refine_cycles=refine_cycles,
                          miller_set=miller_set, multi_index=multi_index,
                          indexed_peak_ids=indexed_peak_ids)
    if solution.match_rate > 0:
      solutions.append(solution)
  return solutions


def index_once(peaks, table, A0, A0_inv, 
               seed_pool_size=5,
               seed_len_tol=3.E7, seed_angle_tol=1.0,
               seed_hkl_tol=0.1, eval_tol=0.25,
               centering="P", centering_factor=0., 
               refine_cycles=10,
               miller_set=None, multi_index=False,
               indexed_peak_ids=[]):
  qs = peaks[:, 4:7]
  unindexed_peak_ids = list(set(range(min(qs.shape[0], seed_pool_size))) - set(indexed_peak_ids))
  seed_pool = list(combinations(unindexed_peak_ids, 2))
  good_solutions = []
  # collect good solutions
  for i in tqdm(range(len(seed_pool))):
    seed = seed_pool[i]
    q1, q2 = qs[seed,:]
    q1_len, q2_len = norm(q1), norm(q2)
    if q1_len < q2_len:
        q1, q2 = q2, q1
        q1_len, q2_len = q2_len, q1_len
    angle = calc_angle(q1, q2)
    match_ids = np.where((np.abs(q1_len - table['LA'][:,0]) < seed_len_tol) * 
               (np.abs(q2_len - table['LA'][:,1]) < seed_len_tol) *
               (np.abs(angle - table['LA'][:,2]) < seed_angle_tol))[0]
    for match_id in match_ids:
      hkl1 = table['hkl1'][match_id]
      hkl2 = table['hkl2'][match_id]
      ref_q1, ref_q2 = A0.dot(hkl1), A0.dot(hkl2)
      solution = Solution()
      solution.R = calc_rotation_matrix(q1, q2, ref_q1, ref_q2)
      solution = eval_solution(solution, qs, A0_inv, eval_tol=eval_tol, 
        centering=centering, centering_factor=centering_factor, 
        miller_set=miller_set, seed=seed, seed_hkl_tol=seed_hkl_tol, 
        indexed_peak_ids=indexed_peak_ids)

      # only keep solution from good seed 
      if solution.seed_error <= seed_hkl_tol:
        good_solutions.append(solution)

  # pick up best solution
  if len(good_solutions) > 0:
    good_solutions.sort(key=lambda x: x.total_score, reverse=True)
    best_score = good_solutions[0].total_score
    best_solutions = [solution for solution in good_solutions if solution.total_score==best_score]
    best_solutions.sort(key=lambda x: x.total_error, reverse=False)
    best_solution = best_solutions[0]  # best solution has highest total score and lowest total error
  else:
    best_solution = None

  # refine best solution if exists
  if best_solution is None:
    dummy_solution = Solution()
    dummy_solution.R = np.identity(3)
    dummy_solution.match_rate = 0.
    return dummy_solution
  else:
    best_solution.A = best_solution.R.dot(A0)
    eXYZs = np.abs(best_solution.A.dot(best_solution.rhkls.T) - qs.T).T  # Fourier space error between peaks and predicted spots
    dists = norm(eXYZs, axis=1)
    best_solution.pair_dist = dists[best_solution.pair_ids].mean()  # average distance between matched peaks and the correspoding predicted spots
    best_solution.A = best_solution.R.dot(A0)  
    refined_solution = refine(best_solution, qs, refine_cycles)  # refine A matrix with matched pairs to minimize norm(AH-q)

  return refined_solution


def refine(solution, qs, refine_cycle):
  A_refined = solution.A.copy()
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
  rhkls = solution.rhkls
  pair_ids = solution.pair_ids
  for i in range(refine_cycle):
    for j in range(len(pair_ids)):  # refine by each reflection
      pair_id = pair_ids[j]
      x0 = A_refined.reshape((-1))
      rhkl = rhkls[pair_id,:]
      q = qs[pair_id,:]
      args = (rhkl[0], rhkl[1], rhkl[2], q[0], q[1], q[2])
      res = fmin_cg(_fun, x0, fprime=_gradient, args=args, disp=0)
      A_refined = res.reshape((3,3))
  eXYZs = np.abs(A_refined.dot(rhkls.T) - qs.T).T
  dists = norm(eXYZs, axis=1)
  pair_dist = dists[pair_ids].mean()

  if pair_dist < solution.pair_dist:
    solution.A_refined = A_refined
    solution.pair_dist_refined = pair_dist
    solution.hkls_refined = np.linalg.inv(A_refined).dot(qs.T).T
    solution.rhkls_refined = np.rint(solution.hkls_refined)
    solution.ehkls_refined = np.abs(solution.hkls_refined - solution.rhkls_refined)
  else:
    solution.A_refined = solution.A.copy()
    solution.pair_dist_refined = solution.pair_dist
    solution.hkls_refined = solution.hkls.copy()
    solution.rhkls_refined = solution.rhkls.copy()
    solution.ehkls_refined = solution.ehkls.copy()
  return solution


if __name__ == '__main__':
  config_file = sys.argv[1]
  # loading configurations
  config = yaml.load(open(config_file))
  cell_parameters = np.asarray(config['cell parameters']) 
  cell_parameters[:3] *= 1E-10  # convert to meters

  detector_distance = config['detector distance']
  pixel_size = config['pixel size'] 

  res_cutoff = config['resolution cutoff'] * 1.E10  # in angstrom
  lattice_type = config['lattice type']
  centering = config['centering']
  table_filepath = config['reference table']

  peak_list_dir = config['peak list directory']
  output_dir = config['output directory']
  sort_by = config['sort by']
  seed_pool_size = int(config['seed pool size'])
  refine_cycles = int(config['refine cycles'])
  seed_len_tol = float(config['seed length tolerance'])
  seed_angle_tol = float(config['seed angle tolerance'])
  seed_hkl_tol = float(config['seed hkl tolerance'])
  centering_factor = float(config['centering factor'])
  if config.has_key('first event'):
    first_event = int(config['first event'])
  else:
    first_event = 0
  if config.has_key('last event'):
    last_event = int(config['last event'])
  else:
    last_event = len(glob.glob(peak_list_dir + '/*.txt')) - 1
  eval_tol = float(config['eval tolerance'])
  multi_index = config['multi index']
  miller_set = None
  if config.has_key('hkl constraint'):
    if config['hkl constraint'] is True:
      hkl_file = config['hkl file']
      if os.path.exists(hkl_file):
        miller_set = np.loadtxt(config['hkl file'])
      else:
        print('ERROR! Must specify a hkl file if set hkl constraint to True!')
        sys.exit()

  # calculate reference transform matrix
  A0 = calc_transform_matrix(cell_parameters,
    lattice_type=lattice_type)
  A0_inv = np.linalg.inv(A0)

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
    logging.info(str(config))
    # assign jobs to slaves and master itself
    nb_patterns = last_event - first_event + 1
    job_size = nb_patterns // size 
    jobs = []
    for i in range(size):
      if i == (size - 1):
        job = np.arange(i*job_size+first_event, last_event+1)
      else:
        job = np.arange(i*job_size+first_event, (i+1)*job_size+first_event)
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
    'spind_rank-%d.txt' % rank), 'w')
  for i in range(len(job)):
    event_id = job[i]
    filename = 'event-%d.txt' % event_id
    filepath = os.path.join(peak_list_dir, filename)
    if not os.path.exists(filepath):
      logging.warning('peak list file %s do not exist' % filepath)
    else:
      logging.info('Rank %d working on event %04d: %s' % 
        (rank, event_id, filepath))
      peaks = load_peaks(filepath, sort_by, res_cutoff)
      solutions = index(
        peaks, table, A0, A0_inv, 
        seed_pool_size=seed_pool_size,
        seed_len_tol=seed_len_tol, seed_angle_tol=seed_angle_tol,
        seed_hkl_tol=seed_hkl_tol, eval_tol=eval_tol,
        centering=centering, centering_factor=centering_factor, 
        refine_cycles=refine_cycles,
        miller_set=miller_set, multi_index=multi_index
      )
      for j in range(len(solutions)):
        solution = solutions[j]
        if solution.match_rate == 0.:
          continue
        # writing basic solution info
        A = solution.A_refined
        output.write('%6d %2d %.2f %4d %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E\n'
               % (event_id, 
                j+1,  # jth crystal
                solution.match_rate, 
                solution.nb_pairs, 
                solution.pair_dist_refined,
                A[0,0], A[1,0], A[2,0],
                A[0,1], A[1,1], A[2,1],
                A[0,2], A[1,2], A[2,2],
                ))
        # writing detailed indexing result
        match_flags = np.zeros((peaks.shape[0], 1))
        match_flags[solution.pair_ids] = 1
        detailed_info = np.hstack((peaks[:,:2], solution.hkls_refined, solution.ehkls_refined, match_flags))
        np.savetxt(os.path.join(output_dir, 
          'spind_event-%d_crystal-%d.txt' % (event_id, j+1)), 
          detailed_info, fmt='%.9e %.9e %.3f %.3f %.3f %.3f %.3f %.3f %d')  

        if solution.match_rate >= 0.5:
          nb_indexed += 1
          indexed_events.append(event_id) 

        print('=' * 40)
        print('event id: %d' % job[i])
        print('crystal id: %d' % (j+1))
        print('total peaks: %d' % len(peaks))
        print('matched peaks: %d' % solution.nb_pairs)
        print('match rate: %.4f' % solution.match_rate)
        print('pair dist: %.3E' % solution.pair_dist)
        print('refined pair dist: %.3E' % solution.pair_dist_refined)
        print('centering score: %.4f' % solution.centering_score)
        print('total score: %.4f' % solution.total_score)
        print('seed error: %.4f' % solution.seed_error)
        print('total error: %.4f' % solution.total_error)
        print('=' * 40)

      logging.info('Rank %d indexing rate: %.2f%%' % (rank, nb_indexed*100./(i+1)))
  print('Rank %d: %d indexed with match rate > 0.5 and centering score > 0.5' % (rank, nb_indexed))
  print('Indexed event %d', indexed_events)
  output.close()

  comm.barrier()
  # merge indexing results to a single txt file
  if rank == 0:
    data = np.array([])
    for i in range(size):
      if i == 0:
        data = np.loadtxt(os.path.join(output_dir, 'spind_rank-0.txt'))
      else:
        data = np.concatenate((data, np.loadtxt(os.path.join(output_dir, 
            'spind_rank-%d.txt' % i))), axis=0)
    data = data.reshape((-1, 14))
    np.savetxt(os.path.join(output_dir, 'spind.txt'),
           data, fmt="%6d %2d %.2f %4d %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E")
    if data.shape[0] == 0:
      overall_indexing_rate = 0.
    else:
      overall_indexing_rate = float((data[:,2] > 0.5).sum()) / float(data.shape[0]) * 100.
    print('Overall indexing rate: %.2f%%' % overall_indexing_rate)