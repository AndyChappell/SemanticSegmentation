import os
import cv2
import csv
import sys
import matplotlib.pyplot as plt
import re
import numpy as np
import glob
from PIL import Image

VOID_CODE = 0
SHOWER_CODE = 1
TRACK_CODE = 2

class Geometry:
    def __init__(self, xuvw_min, xuvw_max):
        self.x_min = xuvw_min[0]
        self.u_min = xuvw_min[1]
        self.v_min = xuvw_min[2]
        self.w_min = xuvw_min[3]
        self.x_max = xuvw_max[0]
        self.u_max = xuvw_max[1]
        self.v_max = xuvw_max[2]
        self.w_max = xuvw_max[3]

    def __repr__(self):
        return "min(XUVW) = ({}, {}, {}, {})   max(XUVW) = ({}, {}, {}, {})".format(
            self.x_min, self.u_min, self.v_min, self.w_min,
            self.x_max, self.u_max, self.v_max, self.w_max)

# DUNE 10kt 1x2x6 MCC11 - Derived from TPC volume
def setup_dune_10kt_1x2x6_geometry():
    return Geometry((-362.662, -353.456, -353.211, -0.876221),
        (362.662, 1484.12, 1484.37, 1393.65))

geometry = setup_dune_10kt_1x2x6_geometry()

#===============================================================================

def make_images(input_file, output_folder = '', image_size = (208, 512)):
    image_height, image_width = image_size
    f = open(input_file, 'r')
    reader = csv.reader(f)
    name = os.path.splitext(os.path.basename(input_file))[0]

    is_u = True if "CaloHitListU" in input_file else False
    is_v = True if "CaloHitListV" in input_file else False
    is_w = True if "CaloHitListW" in input_file else False

#    span = 980
#    x_low = -420
#    x_high = x_low + span
#    z_low = -350 if is_u else 0 if is_v else -25
#    z_high = z_low + span
    x_low, x_high = int(np.floor(geometry.x_min)), int(np.floor(geometry.x_max))
    if is_u:
        z_low, z_high = int(np.floor(geometry.u_min)), int(np.floor(geometry.u_max))
    elif is_v:
        z_low, z_high = int(np.floor(geometry.v_min)), int(np.floor(geometry.v_max))
    else:
        z_low, z_high = int(np.floor(geometry.w_min)), int(np.floor(geometry.w_max))

    x_bin_edges = np.linspace(x_low, x_high, image_height + 1)
    z_bin_edges = np.linspace(z_low, z_high, image_width + 1)

    for i, row in enumerate(reader):
        tempname = name + "_{}".format(i)
        process_picture(row, x_bin_edges, z_bin_edges, image_size, tempname, output_folder)

    f.close()

#===============================================================================

def process_picture(row, x_bin_edges, z_bin_edges, image_size, name, output_folder):
    image_height, image_width = image_size
    x = []
    z = []
    r = []
    g = []
    b = []
    code = []
    row.pop(0) # Date Time
    row.pop(0) # NHits
    row.pop() # 1

    n_elements = 5

    # Check the row contains the correct number of entriess
    if len(row) % n_elements != 0:
        print('Missing information in input file')

    shower_pdg_codes = [11, -11, 22]

    pdg_codes = set([])

    for hit_index in range(int(len(row) / n_elements)):
        # Skip if hit originates from non standard particle
        if 'e' in row[n_elements * hit_index + 3]:
            continue

        x_position_index = n_elements * hit_index + 0
        z_position_index = n_elements * hit_index + 2
        pdg_index = n_elements * hit_index + 3
        nuance_code_index = n_elements * hit_index + 4

        x.append(float(row[x_position_index]))
        z.append(float(row[z_position_index]))
        pdg = int(row[pdg_index])
        pdg_codes.add(pdg)
        nuance_code = int(row[nuance_code_index])

        if pdg in shower_pdg_codes:
            r.append(0)
            g.append(1)
            b.append(0)
            code.append(SHOWER_CODE)
        else:
            r.append(1)
            g.append(0)
            b.append(0)
            code.append(TRACK_CODE)

    x = np.array(x)
    z = np.array(z)

    x_bin_indices = np.digitize(x, x_bin_edges)
    z_bin_indices = np.digitize(z, z_bin_edges)

    # Build input histogram
    input_histogram, x_bin_edges, z_bin_edges = np.histogram2d(x, z, bins = (x_bin_edges, z_bin_edges))
    input_histogram = input_histogram * float(255)

    input_image_name = os.path.join(output_folder, "InputImage_" + name + "_0.png")
    #cv2.imwrite(input_image_name, input_histogram)
    cv2.imwrite(input_image_name, input_histogram)

    # Build truth histogram
    #truth_histogram = np.zeros((image_height, image_width, 3), 'uint8')

    #for idx, x_iter in enumerate(x):
    #    index_x = x_bin_indices[idx]
    #    index_z = z_bin_indices[idx]
    #    if index_x < image_height and index_z < image_width:
    #        truth_histogram[index_x, index_z] = [r[idx]*255, g[idx]*255, b[idx]*255]

    #truth_image_name = os.path.join(output_folder, "TruthImage_" + name + "_0.png")
    # OpenCV writes in BGR, whilst we have constructed RGB, so convert
    #cv2.imwrite(truth_image_name, cv2.cvtColor(truth_histogram, cv2.COLOR_RGB2BGR))

    truth_histogram = np.zeros((image_height, image_width), 'uint8')

    for idx, x_iter in enumerate(x):
        index_x = x_bin_indices[idx]
        index_z = z_bin_indices[idx]
        if index_x < image_height and index_z < image_width:
            truth_histogram[index_x, index_z] = code[idx]

    truth_image_name = os.path.join(output_folder, "TruthImage_" + name + "_0.png")

    cv2.imwrite(truth_image_name, truth_histogram)

#========================================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=str, required=True,
        dest="input_dir", help="The path containing the input text files")
    parser.add_argument("--file-prefix", "-p", type=str, default="DUNEFD_MC11",
        dest="file_prefix", help="The filename prefix")
    parser.add_argument("--output-dir", "-o", type=str, required=True,
        dest="output_dir", help="The path in which output image files will be stored")
    args = parser.parse_args()

    file_pattern = '{}_CaloHitList*.txt'.format(args.file_prefix)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_files = [f for f in glob.glob(os.path.join(args.input_dir, file_pattern))]
    all_files.sort()

    setup_dune_10kt_1x2x6_geometry()

    for filename in all_files:
        print(filename)
        make_images(filename, args.output_dir)
