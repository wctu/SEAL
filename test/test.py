"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
import os
import cv2
import numpy as np
import torch
from ERSModule import *
from network import *


def main():
    # specify gpu id
    gpu_id = 0

    # configurations
    conn8 = 1

    # number of superpixels to be tested
    nC_list = [100, 200, 300, 400, 500, 600]

    # the file list of test data
    img_folder = './input'
    img_fullpath = []
    for filename in os.listdir(img_folder):
        if filename.endswith('.jpg'):
            img_fullpath.append(os.path.join(img_folder, filename))

    print('Found %d images' % (len(img_fullpath)))
    img_fullpath.sort()

    # prepare output folders
    imlog_dir = './output'
    if not os.path.exists(imlog_dir):
        os.makedirs(imlog_dir)

    label_dir = []
    for nC in nC_list:
        tmp_dir = os.path.join(imlog_dir, str(nC))
        label_dir.append(tmp_dir)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

    affinity_dir = os.path.join(imlog_dir, 'affinity')
    if not os.path.exists(affinity_dir):
        os.makedirs(affinity_dir)

    # model
    model = PixelAffinityNet(nr_channel=128, conv1_size=7, use_canny=True)
    model.load_state_dict(torch.load('./bsds500.pkl', map_location=lambda storage, loc: storage))
    model.eval()
    model.cuda(gpu_id)

    print('=== START TESTING ===')
    for i, img_path in enumerate(img_fullpath):
        filename = os.path.basename(img_path)
        basename = filename[0:-4]
        print("%d: %s" % (i, filename))
        image = cv2.imread(img_path)
        h, w, ch = image.shape

        input1 = image.transpose((2, 0, 1))
        input1 = np.float32(input1) / 255.0
        input1 = np.reshape(input1, [1, ch, h, w])
        input1 = torch.from_numpy(input1)

        # compute Canny edges
        edge = cv2.Canny(image, 50, 100)
        edge = 1. - np.float32(edge) / 255.
        edge = np.reshape(edge, [1, 1, h, w])
        input2 = torch.from_numpy(edge)
        inputs = torch.cat((input1, input2), 1)
        inputs = inputs.cuda(gpu_id, non_blocking=True)

        # inference
        with torch.no_grad():
            out_x = model(inputs)
            inputs_t = torch.transpose(inputs, 2, 3)
            out_y_t = model(inputs_t)
            out_y = torch.transpose(out_y_t, 2, 3)
            outputs = torch.cat((out_x, out_y), 1)

        # compute superpixels
        affinity = outputs[0].data.cpu().numpy()
        affinity_list = affinity.flatten().tolist()
        output = np.zeros_like(image)
        for i, nC in enumerate(nC_list):
            print('    nC = %d' % nC)
            sp_list = ERSWgtOnly(affinity_list, h, w, nC, conn8, 0.5)
            sp_label = np.reshape(np.asarray(sp_list), (h, w), order='C')

            # save labels as uint16 png files
            output_fullpath = os.path.join(label_dir[i], basename + '.png')
            cv2.imwrite(output_fullpath, np.uint16(sp_label))

            # save superpixel contour
            np.copyto(output, image)
            for y in range(0, h):
                for x in range(0, w):
                    if (x < w - 1) and (sp_label[y, x] != sp_label[y, x + 1]):
                        output[y, x, :] = [0, 0, 255]
                    if (y < h - 1) and (sp_label[y, x] != sp_label[y + 1, x]):
                        output[y, x, :] = [0, 0, 255]
            output_fullpath = os.path.join(label_dir[i], basename + '_contour.png')
            cv2.imwrite(output_fullpath, output)

        # save affinity maps
        output_affinity = np.uint8(255 * affinity)
        output_fullpath = os.path.join(affinity_dir, basename + '_x.png')
        cv2.imwrite(output_fullpath, output_affinity[0, ::])
        output_fullpath = os.path.join(affinity_dir, basename + '_y.png')
        cv2.imwrite(output_fullpath, output_affinity[1, ::])


if __name__ == '__main__':
    main()
