#!/usr/bin/env python

"""
Usage:
    extract_peaks.py -i cxi_file_or_dir -g geometry_file -d det_dist [options]

Options:
    -h --help                                       Show this screen.
    -i cxi_file_or_dir                              CXI file or directory.
    -g geometry_file                                Geometry file in cheetah, crystfel or psana format.
    -d det_dist                                     Detector distance in meters.
    -p pixel_size                                   Pixel size in meters [default: 110E-6].
    -o output_dir                                   Output directory [default: output].
    --sort-by=<metric>                              Sort peaks by certain metric [default: SNR].
    --res=<resolution>                              Resolution threshold for high priority peaks in angstrom [default: 4.5].
    --output-format=output_format                   Ourput format, multiple plain txt files or single hdf5 files [default: h5].
    --scale                                         Special option for cxij6916 for filtered region scaling.
"""

from docopt import docopt
import os
import h5py
from tqdm import tqdm  # show a progress bar for loops
import numpy as np
from math import sin, atan, sqrt, exp, cos
import glob


h = 4.135667662E-15  # Planck constant in eV*s
c = 2.99792458E8  # light speed in meters per sec


class Peak(object):
    """docstring for peak"""
    def __init__(self, x, y, res, SNR, total_intensity, encoder_value, photon_energy):
        """Summary
        
        Args:
            x (double): x coordinate in assembled detector in pixels.
            y (double): y coordinate in assembled detector in pixels.
            res (double): resolution in angstrom.
            SNR (double): peak signal to noise ratio.
            total_intensity (double): peak total intensity.
            encoder_value (double): camera offset.
            photon_energy (double): photom energy in eV.
        """
        super(Peak, self).__init__()
        self.x = x
        self.y = y
        self.res = res 
        self.SNR = SNR
        self.total_intensity = total_intensity
        self.encoder_value = encoder_value
        self.photon_energy = photon_energy


def parse_peak_list_filename(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    experiment, run_id, class_id = basename.split('-')
    class_id = class_id.split('e')[0]  # remove event id
    return experiment, run_id, class_id


def load_geom(filename):
    print "load %s" % filename
    ext = os.path.splitext(filename)[1]
    if ext == '.h5':
        f = h5py.File(filename, 'r')
        return f['x'].value, f['y'].value, f['z'].value
    elif ext == '.geom':
        from psgeom import camera
        cspad = camera.Cspad.from_crystfel_file(filename)
        cspad.to_cheetah_file('.geom.h5')
        f = h5py.File('.geom.h5', 'r')
        return f['x'].value, f['y'].value, f['z'].value
    elif ext == '.psana':
        from psgeom import camera
        cspad = camera.Cspad.from_psana_file(filename)
        cspad.to_cheetah_file('.geom.h5')
        f = h5py.File('.geom.h5', 'r')
        return f['x'].value, f['y'].value, f['z'].value
    else:
        print('Wrong geometry: %s. You must provide Cheetah, CrystFEL or psana geometry file.')
        return None


def extract_peaks(cxi_file, output_dir, sort_by='SNR', 
                  res_threshold=4.5E-10, scale=False,
                  output_format='h5'):
    data = h5py.File(cxi_file,'r')
    basename, ext = os.path.splitext(os.path.basename(cxi_file))

    # mkdir for output
    if not os.path.isdir(output_dir):
        os.makedirs('%s' % output_dir)

    nb_events = data['entry_1/data_1/data'].shape[0]
    
    print('Processing %s with geometry %s' % (cxi_file, geom_file))
    peak_lists = []
    dataset_names = []
    for event_id in tqdm(range(nb_events)):
        if output_format == 'txt':
            output = output_dir + '/' + basename + '-e%04d.txt' % event_id
        elif output_format == 'h5':
            dataset_names.append(basename + '-e%04d' % event_id)
        nb_peaks = np.nonzero(data['entry_1/result_1/peakTotalIntensity'][event_id][:])[0].size
        peak_list = []
        for peak_id in range(nb_peaks):
            rawX = int(data['/entry_1/result_1/peakXPosRaw'][event_id][peak_id])
            rawY = int(data['/entry_1/result_1/peakYPosRaw'][event_id][peak_id])
            assX = geom_x[rawY][rawX] / pixel_size
            assY = geom_y[rawY][rawX] / pixel_size
        
            total_intensity = data['/entry_1/result_1/peakTotalIntensity'][event_id][peak_id]
            SNR = data['/entry_1/result_1/peakSNR'][event_id][peak_id]
            encoder_value = data['/LCLS/detector_1/EncoderValue'][event_id]
            photon_energy = data['LCLS/photon_energy_eV'][event_id]
            # calculate resolution
            r = sqrt(assX ** 2. + assY ** 2.)
            lam = h * c / photon_energy
            res = lam / (sin(0.5 * atan(r * pixel_size / det_dist)) * 2)
            if scale:
                zn_thickness = 50E-6
                absorption_length = 41.73E-6
                theta = atan(r * pixel_size / det_dist)  # scattering angle
                scaling = exp(zn_thickness / absorption_length / cos(theta))
                total_intensity *= scaling
            peak = Peak(assX, assY, res, SNR, total_intensity, encoder_value, photon_energy)
            peak_list.append(peak)

        if sort_by == 'SNR':
            peak_list.sort(key=lambda peak: peak.SNR, reverse=True)
        elif sort_by == 'total_intensity':
            peak_list.sort(key=lambda peak: peak.total_intensity, reverse=True)
        elif sort_by == 'res':
            peak_list.sort(key=lambda peak: peak.res, reverse=False)
        else:
            print('Unimplemented sort strategy: %s' % sort_by)
            return None

        HP_ids = []  # high priority peak indices 
        LP_ids = []  # low priority peak indices
        for peak_id in range(len(peak_list)):
            peak = peak_list[peak_id]
            if peak.res > res_threshold:
                HP_ids.append(peak_id)
            else:
                LP_ids.append(peak_id)
        # rearange peaks order according to the priority
        new_peak_list = []
        for peak_id in HP_ids:
            new_peak_list.append(peak_list[peak_id])
        for peak_id in LP_ids:
            new_peak_list.append(peak_list[peak_id])
        peak_lists.append(new_peak_list)

        # write to file
        if output_format == 'txt':
            f = open(output, 'w')
            for peak in new_peak_list:
                f.write('%.5e %.5e %.5e %.5e %.5e %.5e %5.2e\n' % 
                        (peak.x, peak.y, peak.total_intensity, 
                         peak.SNR, peak.photon_energy, peak.encoder_value,
                         peak.res))
    if output_format == 'h5':
        output = output_dir + '/' + basename + '.h5'
        f = h5py.File(output)
        for i, peak_list in enumerate(peak_lists):
            peak_array = []
            for peak in peak_list:
                peak_array.append([peak.x, peak.y, peak.total_intensity, 
                         peak.SNR, peak.photon_energy, peak.encoder_value,
                         peak.res])
            f.create_dataset(dataset_names[i], data=np.asarray(peak_array))
        f.close()



if __name__ == '__main__':
    argv = docopt(__doc__)
    cxi_file_or_dir = argv['-i']
    geom_file = argv['-g']
    sort_by = argv['--sort-by']
    res_threshold = float(argv['--res']) * 1.E-10  # convert resolution in angstroms to meters.
    output_dir = argv['-o']
    det_dist = float(argv['-d'])
    pixel_size = float(argv['-p'])
    output_format = argv['--output-format']
    scale = argv['--scale']
    # load geometry
    geom_x, geom_y, geom_z = load_geom(geom_file)

    # load cxi file
    if os.path.isdir(cxi_file_or_dir):  # extract peaks from multiple cxi files
        cxi_files = glob.glob(cxi_file_or_dir + '/' + '*.cxi')
    else:  # extract peaks from single cxi file
        cxi_files = [cxi_file_or_dir, ]
    for cxi_file in cxi_files:
        experiment, run_id, class_id = parse_peak_list_filename(cxi_file)
        extract_peaks(cxi_file, '%s/%s/%s' % (output_dir, run_id, class_id), 
            sort_by=sort_by, res_threshold=res_threshold, 
            scale=scale, output_format=output_format)