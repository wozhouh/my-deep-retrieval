# -*- coding: utf-8 -*-

# usage: python ./myPython/raw_concat_test.py
#   --proto ./proto/raw_concat_resnet101_normpython.prototxt
#   --weights ./caffemodel/deep_image_retrieval_model.caffemodel

import os
import sys
import numpy as np
import caffe
import argparse
from tqdm import tqdm
from oxford_helper import *

if __name__ == '__main__':

    # Configure
    parser = argparse.ArgumentParser(description='Evaluate')
    parser.add_argument('--gpu', type=int, required=False, help='GPU ID to use (e.g. 0)')
    parser.add_argument('--S', type=int, required=False, help='Resize larger side of image to S pixels (e.g. 800)')
    parser.add_argument('--L', type=int, required=False, help='Use L spatial levels (e.g. 2)')
    parser.add_argument('--proto', type=str, required=True, help='Path to the prototxt file')
    parser.add_argument('--weights', type=str, required=True, help='Path to the caffemodel file')
    parser.add_argument('--dataset', type=str, required=False, help='Path to the Oxford / Paris directory')
    parser.add_argument('--dataset_name', type=str, required=False, help='Dataset name')
    parser.add_argument('--eval_binary', type=str, required=False,
                        help='Path to the compute_ap binary to evaluate Oxford / Paris')
    parser.add_argument('--temp_dir', type=str, required=False,
                        help='Path to a temporary directory to store features and scores')
    parser.add_argument('--features_dir', type=str, required=False,
                        help='Path to a temporary directory to store ROI-pooling features and PCA transformation')
    parser.set_defaults(gpu=0)
    parser.set_defaults(S=512)
    parser.set_defaults(L=2)
    parser.set_defaults(dataset_name='Oxford')
    parser.set_defaults(dataset='/home/processyuan/data/Oxford/')
    parser.set_defaults(eval_binary='/home/processyuan/code/NetworkOptimization/deep-retrieval/eval/compute_ap')
    parser.set_defaults(temp_dir='/home/processyuan/code/NetworkOptimization/deep-retrieval/eval/temp/')
    parser.set_defaults(features_dir='/home/processyuan/code/NetworkOptimization/deep-retrieval/features/raw_concat/')
    args = parser.parse_args()
    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)
    S = args.S
    L = args.L

    # Configure caffe and load the network
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # Load the dataset and the image helper
    oxford_dataset = OxfordDataset(args.dataset)
    image_helper = ImageHelper(S, L)

    N_queries = oxford_dataset.N_queries
    N_dataset = oxford_dataset.N_images
    layers = ['rmac_master_32/normalized',
              'rmac_branch_16/normalized',
              'rmac_branch_8/normalized',
              'rmac_branch_4/normalized']
    dim_layers = [net.blobs[layers[k]].data.shape[1] for k in range(len(layers))]
    num_layers = len(layers)
    eps = 1e-8

    features_queries_list = [np.zeros((N_queries, dim_layers[k]), dtype=np.float32) for k in range(num_layers)]
    features_dataset_list = [np.zeros((N_dataset, dim_layers[k]), dtype=np.float32) for k in range(num_layers)]

    # queries: get ROI-pooling features
    for i in tqdm(range(N_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
        I, R = image_helper.prepare_image_and_grid_regions_for_network(oxford_dataset.get_query_filename(i))
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward()
        for k in range(num_layers):
            (features_queries_list[k])[i] = np.squeeze(net.blobs[layers[k]].data)

    # dataset: get ROI-pooling features
    for i in tqdm(range(N_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
        # Load image, process image, get image regions, feed into the network, get descriptor, and store
        I, R = image_helper.prepare_image_and_grid_regions_for_network(oxford_dataset.get_filename(i))

        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward()
        for k in range(num_layers):
            (features_dataset_list[k])[i] = np.squeeze(net.blobs[layers[k]].data) * dim_layers[k]

    # Save the ROI-pooled features from middle layers
    pooled_rois_queries_fname = ["{0}{1}_S{2}_L{3}_layer{4}_ROIpooling_queries.npy".
                                     format(args.features_dir, args.dataset_name, S, L, k) for k in range(num_layers)]
    pooled_rois_dataset_fname = ["{0}{1}_S{2}_L{3}_layer{4}_ROIpooling_dataset.npy".
                                     format(args.features_dir, args.dataset_name, S, L, k) for k in range(num_layers)]
    for k in range(num_layers):
        np.save(pooled_rois_queries_fname[k], features_queries_list[k])
        np.save(pooled_rois_dataset_fname[k], features_dataset_list[k])

    # Concatenation and normalization
    features_queries = np.hstack((features_queries_list[k] for k in range(num_layers)))
    features_queries /= np.sqrt((features_queries * features_queries).sum(axis=1))[:, None]
    features_dataset = np.hstack((features_dataset_list[k] for k in range(num_layers)))
    features_dataset /= np.sqrt((features_dataset * features_dataset).sum(axis=1))[:, None]

    # Compute similarity
    sim = features_queries.dot(features_dataset.T)

    # Score
    oxford_dataset.score(sim, args.temp_dir, args.eval_binary)
