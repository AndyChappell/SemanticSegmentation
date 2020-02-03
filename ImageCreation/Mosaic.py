import os
import cv2
import csv
import sys
import matplotlib.pyplot as plt
import re
import numpy as np
import glob
from PIL import Image

pdg_codes = set([])

VOID_CODE = 0
SHOWER_CODE = 1
TRACK_CODE = 2
NO_RECO_CODE = 3
K_CODE = 321
P_CODE = 2212
E_CODE = 11
MU_CODE = 13
PI_CODE = 211
GAMMA_CODE = 22
SIGMAP_CODE = 3222
SIGMAM_CODE = 3112

make_volatile = False
make_viewable_truth = False
make_pdg_truth = False

palette = {
    VOID_CODE: [0., 0., 0.],
    SHOWER_CODE: [0., 0., 255.],
    TRACK_CODE: [255., 0., 0.],
    NO_RECO_CODE: [0., 255., 255.],
    K_CODE: [127., 201, 127.],
    P_CODE: [212., 174., 190.],
    E_CODE: [134., 192., 253.],
    MU_CODE: [153., 255., 255.],
    PI_CODE: [128., 108., 56.],
    GAMMA_CODE: [127., 2., 240.],
    SIGMAP_CODE: [23., 91., 191.],
    SIGMAM_CODE: [23., 91., 191.]
}

def fill_pixel(histogram, code, nx, nz, local_x, local_z):
    histogram[nx, nz, local_x, local_z,] = palette[code]

#===============================================================================

def make_images(input_file, output_folder = '', image_size = (208, 512)):
    image_height, image_width = image_size
    f = open(input_file, 'r')
    reader = csv.reader(f)
    name = os.path.splitext(os.path.basename(input_file))[0]

    is_u = True if "CaloHitListU" in input_file else False
    is_v = True if "CaloHitListV" in input_file else False
    is_w = True if "CaloHitListW" in input_file else False

    # Need to get x_low/high and z_low/high from input hits
    #x_bin_edges = np.linspace(x_low, x_high, image_height + 1)
    #z_bin_edges = np.linspace(z_low, z_high, image_width + 1)

    min_dx, min_dz = np.Inf, np.Inf
    x_pixels_max = -np.Inf
    z_pixels_max = -np.Inf
    for i, row in enumerate(reader):
        do_print = (i + 1) % 50 == 0
        if do_print:
            print(f"Event {i + 1}... ", end='')
        tempname = name + "_{}".format(i)
        process_hits(row, tempname, output_folder)
        if do_print:
            print("Processed.")
    f.close()
    print("PDG codes found")
    for val in pdg_codes:
        print(val)


#===============================================================================

def process_hits(hits, name, output_folder, image_size=(256, 256)):
    truth_output_folder = output_folder + "/Images/Truth"
    hits_output_folder = output_folder + "/Images/Hits"
    n_elements = 5
    block_size = 128.0
    num_hits = len(hits) // n_elements
    x = np.zeros(num_hits)
    z = np.zeros(num_hits)
    pdg = np.zeros(num_hits, dtype=np.int32)
    tag = np.zeros(num_hits, dtype=np.int8)
    hits.pop(0) # Date Time
    hits.pop(0) # NHits
    hits.pop() # 1

    # Check the row contains the correct number of entriess
    if len(hits) % n_elements != 0:
        print('Missing information in input file')

    shower_codes = [11, -11, 22]
    if len(hits) == 0: # No reconstructable hits, skip
        return
    for i, hit_index in enumerate(range(len(hits) // n_elements)):
        # Skip if hit originates from non standard particle
        if 'e' in hits[n_elements * hit_index + 3]: continue
        offset = n_elements * hit_index
        x_idx, z_idx, pdg_idx = offset, offset + 2, offset + 3
        reco_idx = offset + 4
        # Nuance code is offset + 4
        x[i] = (float(hits[x_idx]))
        z[i] = (float(hits[z_idx]))
        pdg[i] = (int(hits[pdg_idx]))
        pdg_codes.add(pdg[i])
        is_reconstructable = int(hits[reco_idx])
        if not is_reconstructable:
            tag[i] = NO_RECO_CODE
            continue
        if pdg[i] in shower_codes:
            tag[i] = SHOWER_CODE
        else:
            tag[i] = TRACK_CODE

    image_height, image_width = image_size
    x_min, x_max = np.amin(x), np.amax(x)
    z_min, z_max = np.amin(z), np.amax(z)
    x_range = x_max - x_min
    z_range = z_max - z_min
    n_x = int(np.ceil(x_range / block_size))
    # Need to add 1 for ranges exactly matching block size and for zero range
    if (x_range % block_size) == 0:
        n_x += 1
    n_z = int(np.ceil(z_range / block_size))
    if (z_range % block_size) == 0:
        n_z += 1
    x_bin_edges = np.linspace(x_min, x_min + n_x * block_size, image_width * n_x + 1)
    z_bin_edges = np.linspace(z_min, z_min + n_z * block_size, image_height * n_z + 1)

    truth_histogram = np.zeros((n_x, n_z, image_height, image_width), 'uint8')
    input_histogram = np.zeros((n_x, n_z, image_height, image_width), 'uint8')
    # Output human viewable truth images for debugging
    if make_viewable_truth:
        truth_viewable_histogram = np.zeros((n_x, n_z, image_height, image_width, 3), 'uint8')
    for idx, x_iter in enumerate(x):
        nx = int((x[idx] - x_min) / block_size)
        nz = int((z[idx] - z_min) / block_size)
        dx = (x[idx] - x_min) % block_size
        dz = (z[idx] - z_min) % block_size
        local_x = int(np.floor(dx * image_height / block_size))
        local_z = int(np.floor(dz * image_width / block_size))
        if tag[idx] != NO_RECO_CODE:
            truth_histogram[nx, nz, local_x, local_z] = tag[idx]
            input_histogram[nx, nz, local_x, local_z] = 255.0
        if make_pdg_truth:
            fill_pixel(truth_viewable_histogram, abs(pdg[idx]), nx, nz, local_x, local_z)
        elif make_viewable_truth:
            fill_pixel(truth_viewable_histogram, tag[idx], nx, nz, local_x, local_z)

    for i in range(n_x):
        for j in range(n_z):
            if np.all(truth_histogram[i,j,...] == 0): continue

            # Produce tiling images for debug
            if make_volatile:
                if i < (n_x - 1) and j < (n_z - 1):
                    if not np.all(truth_histogram[i,j+1,...] == 0) and not np.all(truth_histogram[i+1,j,...] == 0) and not np.all(truth_histogram[i+1,j+1,...] == 0):
                        volatile_histogram_tl = np.copy(input_histogram[i,j,...])
                        volatile_histogram_bl = np.copy(input_histogram[i,j+1,...])
                        volatile_histogram_tr = np.copy(input_histogram[i+1,j,...])
                        volatile_histogram_br = np.copy(input_histogram[i+1,j+1,...])
                        for hist in (volatile_histogram_tl, volatile_histogram_bl, volatile_histogram_tr, volatile_histogram_br):
                            hist[0,:] = 128.0
                            hist[-1,:] = 128.0
                            hist[:,0] = 128.0
                            hist[:,-1] = 128.0
                        volatile_image_name = os.path.join(hits_output_folder, f"Volatile_{name}_{i}_{j}.png")
                        cv2.imwrite(volatile_image_name, volatile_histogram_tl)
                        volatile_image_name = os.path.join(hits_output_folder, f"Volatile_{name}_{i}_{j+1}.png")
                        cv2.imwrite(volatile_image_name, volatile_histogram_bl)
                        volatile_image_name = os.path.join(hits_output_folder, f"Volatile_{name}_{i+1}_{j}.png")
                        cv2.imwrite(volatile_image_name, volatile_histogram_tr)
                        volatile_image_name = os.path.join(hits_output_folder, f"Volatile_{name}_{i+1}_{j+1}.png")
                        cv2.imwrite(volatile_image_name, volatile_histogram_br)

            if np.random.uniform(0.0, 1.0) > 0.2: continue
            truth_image_name = os.path.join(truth_output_folder, f"Image_{name}_{i}_{j}.png")
            cv2.imwrite(truth_image_name, truth_histogram[i, j, ...])
            input_image_name = os.path.join(hits_output_folder, f"Image_{name}_{i}_{j}.png")
            cv2.imwrite(input_image_name, input_histogram[i, j, ...])
            if make_viewable_truth:
                truth_image_name = os.path.join(output_folder, f"Images/Viewable/Image_{name}_{i}_{j}.png")
                cv2.imwrite(truth_image_name, truth_viewable_histogram[i, j, ...])

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
    parser.add_argument("--tile", action='store_true')
    parser.add_argument("--truth", action='store_true')
    parser.add_argument("--pdg", action='store_true')
    args = parser.parse_args()

    if args.tile: make_volatile = True
    if args.truth: make_viewable_truth = True
    if args.pdg:
        make_pdg_truth = True
        make_viewable_truth = True

    np.random.seed(42)

    file_pattern = '{}_CaloHitList*.txt'.format(args.file_prefix)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir + "/Images/Truth"):
        os.makedirs(args.output_dir + "/Images/Truth")
    if not os.path.exists(args.output_dir + "/Images/Hits"):
        os.makedirs(args.output_dir + "/Images/Hits")
    if not os.path.exists(args.output_dir + "/Images/Viewable"):
        os.makedirs(args.output_dir + "/Images/Viewable")

    all_files = [f for f in glob.glob(os.path.join(args.input_dir, file_pattern))]
    all_files.sort()

    #setup_dune_10kt_1x2x6_geometry()

    for filename in all_files:
        #if "ListW" in filename: continue
        print(filename)
        make_images(filename, args.output_dir)
