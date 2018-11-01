# -*- coding: utf-8 -*-

# Python script that converts the images in the dataset into feature vectors (numpy array) saved in 'features.npy'
# and generates a .txt file indicating the image filename correspond to which row of the numpy array
# Note test the deployed network should generate ROIs itself

'''
usage:
python ./myPython/cover_convert_traingset.py \
    --proto ./proto/train-distilling/deploy_resnet101_teacher.prototxt \
    --weights ./caffemodel/deep_image_retrieval_model_distilling.caffemodel \
    --img_dir ~/data/cover/training/img/ \
    --features_npy ~/data/cover/training/ \
    --features_txt ~/data/cover/training/training.txt
'''


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
    parser.add_argument('--features_npy', type=str, required=False,
                        help='Path to save the features')
    parser.add_argument('--features_txt', type=str, required=False,
                        help='Path to the file to record the feature index')
    parser.set_defaults(gpu=0)
    parser.set_defaults(L=2)
    parser.set_defaults(proto='/home/processyuan/NetworkOptimization/deep-retrieval/proto/'
                              'distilling/deploy_resnet101_teacher.prototxt')
    parser.set_defaults(weights='/home/processyuan/NetworkOptimization/deep-retrieval/'
                                'caffemodel/deploy_resnet101_teacher.caffemodel')
    parser.set_defaults(img_dir='/home/processyuan/NetworkOptimization/cover/training/img')
    parser.set_defaults(features_npy='/home/processyuan/NetworkOptimization/cover/training/features.npy')
    parser.set_defaults(features_txt='/home/processyuan/NetworkOptimization/cover/training/training.txt')

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

    # convert the images into features vectors by ResNet-101 and R-MAC and save them into numpy array
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

    # save the features and write the txt file
    features_fname = args.features_npy
    np.save(features_fname, features)
    random.shuffle(f_lines)
    for line in f_lines:
        f_txt.write(line)
    f_txt.close()
