# -*- coding: utf-8 -*-

# Python script that converts the cover images into features and labels then save them locally (for test)

import os
import sys
import numpy as np
import caffe
import cv2
import argparse
from tqdm import tqdm
import region_generator as rg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts the images into features save as .npy')
    parser.add_argument('--gpu', type=int, required=False, help='GPU ID to use (e.g. 0)')
    parser.add_argument('--L', type=int, required=False, help='Use L spatial levels (e.g. 2)')
    parser.add_argument('--proto', type=str, required=True, help='Path to the prototxt file')
    parser.add_argument('--weights', type=str, required=True, help='Path to the caffemodel file')
    parser.add_argument('--end', type=str, required=False, help='Define the output layer of the net')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Path to the directory to images')
    parser.add_argument('--features_npy', type=str, required=True,
                        help='Path to save the features')
    parser.add_argument('--features_txt', type=str, required=True,
                        help='Path to the file to record the feature index')
    parser.set_defaults(gpu=0)
    parser.set_defaults(L=2)
    parser.set_defaults(end='rmac/normalized')
    args = parser.parse_args()

    # Configure caffe and load the model
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    means = np.array([103.93900299, 116.77899933, 123.68000031], dtype=np.float32)[None, :, None, None]

    # Output of the model
    output_layer = args.end  # suppose that the layer name is always the same as the blob name
    dim_features = net.blobs[output_layer].data.shape[1]
    images = os.listdir(args.img_dir)
    features = np.zeros((len(images), dim_features), dtype=np.float32)
    f_txt = open(args.features_txt, 'w')
    img_idx = 0

    # convert the images into features vectors by ResNet-101 and R-MAC and save them into numpy array
    for img_file in tqdm(images, file=sys.stdout, leave=False, dynamic_ncols=True):
        # load the image
        img_path = os.path.join(args.img_dir, img_file)
        img_temp = cv2.imread(img_path).transpose(2, 0, 1)
        img = np.expand_dims(img_temp, axis=0) - means
        # get the ROI
        all_regions = [rg.get_rmac_region_coordinates(img_temp.shape[1], img_temp.shape[2], args.L)]
        R = rg.pack_regions_for_network(all_regions)
        net.blobs['data'].reshape(img.shape[0], int(img.shape[1]), int(img.shape[2]), int(img.shape[3]))
        net.blobs['data'].data[:] = img
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end=output_layer)
        features[img_idx, :] = np.squeeze(net.blobs[output_layer].data)
        # write the label into file
        label = str(img_idx) + '\t' + img_file.split('.jpg')[0] + '\n'
        f_txt.write(label)
        img_idx += 1

    # save the features and write the txt file
    np.save(args.features_npy, features)
    f_txt.close()
