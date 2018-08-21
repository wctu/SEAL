# ------------------------------------------------------------------------------
# Author: Wei-Chih Tu
# Function: This script evaluates superpixel ASA and BR for BSDS500 dataset.
# All label maps are expected to be 16-bit single-channel png images.
# Multiple superpixel numbers can be evaluated at the same time.
# It is helpful when evaluating a large dataset.
# ------------------------------------------------------------------------------
import os
import time
import cv2
import multiprocessing
from joblib import Parallel, delayed
from EvalSPModule import *

gtseg_dir = '../data/groundtruth'
label_root_dir = '../data/output'

# tolerence parameter for BR score
r = 1


def evaluate(label_dir):
    mean_asa = 0
    mean_br = 0
    nr_sample = 0
    for root, dirs, files in os.walk(gtseg_dir):
        for filename in files:
            if filename.endswith('.png'):
                gtseg = cv2.imread(os.path.join(gtseg_dir, filename), -1)   # use -1 to read 16-bit png
                label = cv2.imread(os.path.join(label_dir, filename[0:-6]+'.png'), -1)
                h, w = label.shape
                
                label_list = label.flatten().tolist()
                gtseg_list = gtseg.flatten().tolist()
                asa = computeASA(label_list, gtseg_list, 0)
                br = computeBR(label_list, gtseg_list, h, w, r)
                mean_asa += asa
                mean_br += br
                nr_sample += 1

    if nr_sample > 0:
        mean_asa /= nr_sample
        mean_br /= nr_sample

    return mean_asa, mean_br


def main():
    if not os.path.exists(label_root_dir):
        print('%s does not exist' % label_root_dir)
        return

    nC_list = [100, 200, 300, 400, 500, 600]
    print(nC_list)

    num_cores = multiprocessing.cpu_count()
    print('%d cpu cores found' % num_cores)

    # Start evaluation
    tic = time.time()
    
    asa_list = []
    br_list = []
    
    output = Parallel(n_jobs=num_cores)(delayed(evaluate)(os.path.join(label_root_dir, str(nC))) for nC in nC_list)

    for i in range(0, len(nC_list)):
        asa_list.append(output[i][0])
        br_list.append(output[i][1])

    toc = time.time()

    print('ASA:')
    for i in range(0, len(nC_list)):
        print(asa_list[i])

    print('BR:')
    for i in range(0, len(nC_list)):
        print(br_list[i])

    print('Elapsed time = %f sec.' % (toc - tic))

    
if __name__ == '__main__':
    main()
    



