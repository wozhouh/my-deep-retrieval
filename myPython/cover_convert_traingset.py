# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import caffe
import cv2
import argparse
from tqdm import tqdm
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate on the cover dataset')
    parser.add_argument('--gpu', type=int, required=False, help='GPU ID to use (e.g. 0)')
    parser.add_argument('--L', type=int, required=False, help='Use L spatial levels (e.g. 2)')
    parser.add_argument('--proto', type=str, required=False, help='Path to the prototxt file')
    parser.add_argument('--weights', type=str, required=False, help='Path to the caffemodel file')
    parser.add_argument('--img_dir', type=str, required=False,
                        help='Path to the directory to images')
    parser.add_argument('--features_dir', type=str, required=False,
                        help='Path to the directory to save the features')
    parser.add_argument('--features_txt', type=str, required=False,
                        help='Path to the file to record the feature index')
    parser.set_defaults(gpu=0)
    parser.set_defaults(L=2)
    parser.set_defaults(proto='/home/processyuan/NetworkOptimization/deep-retrieval/proto/'
                              'train-distilling/resnet101_teacher.prototxt')
    parser.set_defaults(weights='/home/processyuan/NetworkOptimization/deep-retrieval/'
                                'caffemodel/deep_image_retrieval_model_distilling.caffemodel')
    parser.set_defaults(img_dir='/home/processyuan/NetworkOptimization/cover/training/img')
    parser.set_defaults(features_dir='/home/processyuan/NetworkOptimization/cover/training/')
    parser.set_defaults(features_txt='/home/processyuan/NetworkOptimization/cover/training/train.txt')

    args = parser.parse_args()

    # Configure caffe and load the network ResNet-101
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # Output of ResNet-101
    output_layer = 'rmac/eltwise/normalized'  # suppose that the layer name is always the same as the blob name
    dim_features = net.blobs[output_layer].data.shape[1]
    images = os.listdir(args.img_dir)
    features = np.zeros((len(images), dim_features), dtype=np.float32)
    f_txt = open(args.features_txt, 'w')
    img_idx = 0
    f_lines = []

    for img_file in tqdm(images, file=sys.stdout, leave=False, dynamic_ncols=True):
        img_path = os.path.join(args.img_dir, img_file)
        img_temp = cv2.imread(img_path).transpose(2, 0, 1)
        img = np.expand_dims(img_temp, axis=0)
        net.blobs['data'].reshape(img.shape[0], int(img.shape[1]), int(img.shape[2]), int(img.shape[3]))
        net.blobs['data'].data[:] = img
        net.forward(end=output_layer)
        features[img_idx, :] = np.squeeze(net.blobs[output_layer].data)
        label = img_file + ' ' + str(img_idx) + '\n'
        f_lines.append(label)
        img_idx += 1

    features_fname = os.path.join(args.features_dir, 'features.npy')
    np.save(features_fname, features)
    random.shuffle(f_lines)
    for line in f_lines:
        f_txt.write(line)
    f_txt.close()
