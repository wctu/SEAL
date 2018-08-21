# ------------------------------------------------------------------------------
# Author: Wei-Chih Tu
# Function: This script evaluates superpixel ASA and BR for the BSDS500 dataset.
# All label maps are expected to be 16-bit single-channel png images.
# ------------------------------------------------------------------------------
import os
import time
import cv2
from EvalSPModule import *

gtseg_dir = '../data/groundtruth'
label_dir = '../data/output/500'


def evaluate(gtseg_dir, label_dir):
    mean_asa = 0
    mean_br = 0
    nr_sample = 0
    for root, dirs, files in os.walk(gtseg_dir):
        for filename in files:
            if filename.endswith('.png'):
                gtseg = cv2.imread(os.path.join(gtseg_dir, filename), -1)   # use -1 to read 16-bit png
                label = cv2.imread(os.path.join(label_dir, filename[0:-6]+'.png'), -1)  # modify this if using different naming rule
                h, w = label.shape
                
                label_list = label.flatten().tolist()
                gtseg_list = gtseg.flatten().tolist()
                asa = computeASA(label_list, gtseg_list, 0)
                br = computeBR(label_list, gtseg_list, h, w, 1)
                mean_asa += asa
                mean_br += br
                nr_sample += 1
                # print('%04d: asa = %f, br = %f, t = %f' % (nr_sample, asa, br, toc-tic))

    if nr_sample > 0:
        mean_asa /= nr_sample
        mean_br /= nr_sample

    return mean_asa, mean_br


def main():
    if not os.path.exists(label_dir):
        print('%s does not exist' % label_dir)
        return

    tic = time.time()
    asa, br = evaluate(gtseg_dir, label_dir)
    toc = time.time()

    print('ASA: %f' % asa)
    print('BR: %f' % br)
    print('Elapsed time = %f sec.' % (toc - tic))


if __name__ == '__main__':
    main()
    



