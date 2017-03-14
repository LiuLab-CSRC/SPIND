#!/bin/env python
"""
Usage:
    SPIND.py -i peak_list_dir -t table_file [options]

Options:
    -h --help                                       Show this screen.
    -i peak_list_dir                                Peak list directory.
    -t table_file                                   Table filepath containing spot vector length and pair angles.
    -o output_dir                                   Output directory [default: output].
    --start=<start_event_id>                        The first event id to index [default: 0].
    --end=<end_event_id>                            The last event id to index [default: last].
    --pair-tol=<pair_tol>                           Reciprocal vector length and angle tolerence in pair matching [default: 3.E-3,1.0].
    --eval-tol=<eval_tol>                           HKL tolerence between observed peaks and predicted spots [default: 0.25].
"""

from docopt import docopt
import logging
from mpi4py import MPI
import os
import glob
import numpy as np
import h5py
from math import acos, pi, cos, sin
from numpy.linalg import norm


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
    table = np.asarray(table['TAB'].value, dtype=np.float32).T
    table[:,9] = np.rad2deg(np.arccos(table[:,9]))
    return table


def det2fourier(det_x, det_y, wave_length, detector_distance):
    """Detector 2d coordinates to fourier 3d coordinates
    
    Args:
        det_x (float): Coordinate at x axis in meters on detector
        det_y (float): Coordinate at y axis in meters on detector
    
    Returns:
        TYPE: 3d fourier coordinates
    """
    q1 = np.asarray([det_x, det_y, detector_distance])
    q1 = q1 / np.linalg.norm(q1)
    q0 = np.asarray([0., 0., 1.])
    q = 1. / wave_length * (q1 - q0) * 1.E-10  # in per angstrom
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
    
    Args:
        q1 (TYPE): Description
        q2 (TYPE): Description
        ref_q1 (TYPE): Description
        ref_q2 (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    ref_nv = np.cross(ref_q1, ref_q2) 
    q_nv = np.cross(q1, q2)
    axis = np.cross(ref_nv, q_nv)
    angle = rad2deg(acos(ref_nv.dot(q_nv) / (norm(ref_nv) * norm(q_nv))))
    R1 = axis_angle_to_rotation_matrix(axis, angle)
    rot_ref_q1, rot_ref_q2 = R1.dot(ref_q1), R1.dot(ref_q2)  # rotate ref_q1,2 plane to q1,2 plane

    angle1 = rad2deg(acos(q1.dot(rot_ref_q1) / (norm(rot_ref_q1) * norm(q1))))
    angle2 = rad2deg(acos(q2.dot(rot_ref_q2) / (norm(rot_ref_q2) * norm(q2))))
    angle = (angle1 + angle2) / 2.
    axis = np.cross(rot_ref_q1, q1)
    R2 = axis_angle_to_rotation_matrix(axis, angle)

    R = R2.dot(R1)
    return R


def axis_angle_to_rotation_matrix(axis, angle):
    """Convert axis angle to rotation matrix
    
    Args:
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


def eval_solution(R, qs, A_inv, eval_tol=0.25):
    """Calculate match rate for the given rotation matrix
    
    Args:
        R (TYPE): Rotation matrix
        qs (TYPE): Peak coorinates in fourier space
        A_inv (TYPE): Description
        eval_tol (float, optional): HKL tolerence
    
    Returns:
        TYPE: Description
    """
    R_inv = np.linalg.inv(R)
    qs = np.asarray(qs)
    HKL = A_inv.dot(R_inv.dot(qs.T)).T
    rHKL = np.rint(HKL)
    eHKL = np.abs(HKL - rHKL)
    hit = np.max(eHKL, axis=1) < eval_tol
    score = float(hit.sum()) / len(qs)
    return score


def index(peaks, table, A, A_inv, pair_tol=[3.E-3, 1.0], eval_tol=0.25):
    """Summary
    
    Args:
        peaks (TYPE): Description
        table (numpy.ndarray): Description
        A (TYPE): transform_matrix
        A_inv (TYPE): Description
        pair_tol (list, optional): Description
        eval_tol (float, optional): Description
    
    Returns:
        TYPE: Description
    """
    max_score = 0.
    best_R = np.identity(3)
    qs = []  # q vectors 
    for i in range(len(peaks)):
        peak = peaks[i]
        q = det2fourier(peak[0]*pixel_size, peak[1]*pixel_size, wave_length, detector_distance)
        qs.append(q)
    pair_pool = [[0,1], [0,2], [0,3], [0,4], [1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]
    for i in range(len(pair_pool)):
        pair = pair_pool[i]
        q1, q2 = qs[pair[0]], qs[pair[1]]
        q1_norm, q2_norm = norm(q1), norm(q2)
        angle = rad2deg(acos(q1.dot(q2.T) / (q1_norm * q2_norm)))
        match_ids = np.where((np.abs(q1_norm - table[:,6]) < pair_tol[0]) * 
                             (np.abs(q2_norm - table[:,7]) < pair_tol[0]) *
                             (np.abs(angle - table[:,9]) < pair_tol[1]))[0]
        for match_id in match_ids:
            hkl1, hkl2 = table[match_id][0:3], table[match_id][3:6]
            ref_q1, ref_q2 = A.dot(hkl1), A.dot(hkl2)

            R = calc_rotation_matrix(q1, q2, ref_q1, ref_q2)
            score = eval_solution(R, qs, A_inv, eval_tol=eval_tol)
            if score > max_score:
                max_score = score
                best_R = R
    return max_score, best_R


def calc_abc_star(R, A):
    """Calulate abc_star 
    
    Args:
        R (TYPE): Rotation matrix
        A (TYPE): Trasform matrix
    
    Returns:
        TYPE: Description
    """
    abc_star = R.dot(A)
    a_star, b_star, c_star = abc_star[:,0], abc_star[:,1], abc_star[:,2]
    return a_star, b_star, c_star


if __name__ == '__main__':
    # experiment parameters
    wave_length = 1.306098E-10  # in meters
    detector_distance = 136.4028E-3  # in meters
    pixel_size = 110.E-6
    cell_parameters = [103.45,50.28,69.380,90.00,109.67,90.00]  # in angstroms and degrees
    A = calc_transform_matrix(cell_parameters)
    A_inv = np.linalg.inv(A)

    # parse args
    args = docopt(__doc__)
    peak_list_dir = args['-i']
    experiment, run_id, class_id = parse_peak_list_filename(glob.glob(peak_list_dir + '/*.txt')[0])
    prefix = experiment + '-' + run_id + '-' + class_id
    table_filepath = args['-t']
    output_dir = args['-o']
    start_id = int(args['--start'])
    end_id = args['--end']
    if end_id == 'last':
        end_id = len(glob.glob(peak_list_dir + '/*.txt')) - 1
    else:
        end_id = int(end_id)
    pair_tol_str = args['--pair-tol']
    pair_tol_list = pair_tol_str.split(',')
    pair_tol = np.asarray(pair_tol_list, dtype=np.float)
    eval_tol = float(args['--eval-tol'])

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
        logging.basicConfig(filename=os.path.join(output_dir, 'debug-%d.log' % rank),level=logging.DEBUG)
        logging.info(str(args))
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
        logging.basicConfig(filename=os.path.join(output_dir, 'debug-%d.log' % rank),level=logging.DEBUG)
        logging.info('Rank %d receive job: %s' % (rank, str(job)))

    # workers do assigned jobs
    logging.info('Rank %d processing jobs: %s' % (rank, str(job)))
    table = load_table(table_filepath)
    nb_indexed = 0
    indexed_events = []
    output = open(os.path.join(output_dir, 'spind_indexing-%d.txt' % rank), 'w')
    for i in range(len(job)):
        event_id = job[i]
        filename = prefix + '-e%04d' % event_id + '.txt'
        filepath = os.path.join(peak_list_dir, filename)
        if not os.path.exists(filepath):
            logging.warning('peak list file %s do not exist' % filepath)
        else:
            logging.info('Rank %d working on event %04d: %s' % (rank, event_id, filepath))
            peaks = load_peaks(filepath)
            score, R = index(peaks, table, A, A_inv, pair_tol=pair_tol, eval_tol=eval_tol)
            a_star, b_star, c_star = calc_abc_star(R, A)
            logging.info('Event %04d, score %.2f' % (event_id, score))
            logging.info('abc star: %s' % str(R.dot(A)*10.))
            _c = 1E10  # convert to per meter
            output.write('%6d %.2f %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E\n'
                         % (job[i], score, a_star[0]*_c, a_star[1]*_c, a_star[2]*_c,
                                      b_star[0]*_c, b_star[1]*_c, b_star[2]*_c,
                                      c_star[0]*_c, c_star[1]*_c, c_star[2]*_c))
            if score >= 0.5:
                nb_indexed += 1
                indexed_events.append(event_id)
            print('Event %d nb_peak %d,  match score %f' % (job[i], len(peaks), score))
        logging.info('Rank %d indexing rate: %.2f%%' % (rank, nb_indexed*100/(i+1)))
    print('Rank %d has %d indexed: %s with match score higher than 50%%' % (rank, nb_indexed, str(indexed_events)))
    output.close()

    comm.barrier()
    # merge indexing results to single txt file
    if rank == 0:
        data = np.array([])
        for i in range(size):
            if i == 0:
                data = np.loadtxt(os.path.join(output_dir, 'spind_indexing-0.txt'))
            else:
                data = np.concatenate((data, np.loadtxt(os.path.join(output_dir, 'spind_indexing-%d.txt' % i))), axis=0)
        np.savetxt(os.path.join(output_dir, prefix + '-spind.txt'),
                   data, fmt="%6d %.2f %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E %.4E")
        overall_indexing_rate = float((data[:,1] > 0.5).sum()) / float(data.shape[0])
        logging.info('Overall indexing rate: %.2f%%' % overall_indexing_rate)