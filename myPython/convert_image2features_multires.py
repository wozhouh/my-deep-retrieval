# -*- coding: utf-8 -*-

# Python script that converts the images in the dataset into feature vectors (numpy array) saved in 'features.npy'
# and generates a .txt file indicating the image filename correspond to which row of the numpy array
# Note that the deployed network will not generate ROIs itself

import os
import numpy as np
import caffe
import cv2
import argparse
import random
import region_generator as rg

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

    args = parser.parse_args()

    # Configure caffe and load the network ResNet-101
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # Output of ResNet-101
    output_layer = 'rmac/normalized'
    dim_features = net.blobs[output_layer].data.shape[1]
    images = os.listdir(args.img_dir)
    features = np.zeros((len(images), dim_features), dtype=np.float32)
    f_lines = []
    Ss = [256, 512, 768]
    means = np.array([103.93900299, 116.77899933, 123.68000031], dtype=np.float32)[None, :, None, None]

    # convert the images into features vectors by ResNet-101 and R-MAC and save them into numpy array
    for l, img_file in enumerate(images):
        img_path = os.path.join(args.img_dir, img_file)
        im = cv2.imread(img_path)
        im_size_hw = np.array(im.shape[0:2])
        features_temp = []
        for k, S in enumerate(Ss):
            ratio = float(S) / np.max(im_size_hw)
            new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
            im_resized = cv2.resize(im, (new_size[1], new_size[0]))
            # Transpose for network and subtract mean
            I = im_resized.transpose(2, 0, 1) - means
            all_regions = [rg.get_rmac_region_coordinates(new_size[0], new_size[1], args.L)]
            R = rg.pack_regions_for_network(all_regions)
            net.blobs['data'].reshape(I.shape[0], int(I.shape[1]), int(I.shape[2]), int(I.shape[3]))
            net.blobs['data'].data[:] = I
            net.blobs['rois'].reshape(R.shape[0], R.shape[1])
            net.blobs['rois'].data[:] = R.astype(np.float32)
            net.forward(end=output_layer)
            features_temp.append(np.squeeze(net.blobs[output_layer].data))
        features_sum = np.dstack(features_temp).sum(axis=2)
        features[l, :] /= np.sqrt((features_sum * features_sum).sum(axis=1))
        label = img_file + ' ' + str(l) + '\n'
        f_lines.append(label)
        print("Finished converting %s image(s)" % l)

    # save the features and write the txt file
    features_fname = args.features_npy
    np.save(features_fname, features)
    random.shuffle(f_lines)
    f_txt = open(args.features_txt, 'w')
    for line in f_lines:
        f_txt.write(line)
    f_txt.close()
