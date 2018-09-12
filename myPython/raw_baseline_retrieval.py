# -*- coding: utf-8 -*-

# Raw features of ResNet-101 master output

# usage:
# ./myPython/baseline_features.py
#   --proto ./proto/deploy_resnet101_normpython.prototxt
#   --weights ./caffemodel/deep_image_retrieval_model.caffemodel
#   --temp_dir ./eval/eval_test/

import os
import sys
import numpy as np
import caffe
import argparse
from tqdm import tqdm
from class_helper import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Oxford / Paris')
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
    parser.set_defaults(gpu=0)
    parser.set_defaults(S=512)
    parser.set_defaults(L=2)
    parser.set_defaults(dataset_name='Oxford')
    parser.set_defaults(dataset='/home/processyuan/data/Oxford/')
    parser.set_defaults(eval_binary='/home/processyuan/NetworkOptimization/deep-retrieval/eval/compute_ap')
    parser.set_defaults(temp_dir='/home/processyuan/NetworkOptimization/deep-retrieval/eval/eval_test/')
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
    dataset = Dataset(args.dataset, args.eval_binary)
    image_helper = ImageHelper(S, L)

    N_queries = dataset.N_queries
    N_dataset = dataset.N_images
    dim_master = net.blobs['pooled_rois/normalized_flat'].data.shape[1]
    eps = 1e-8

    features_master_queries = np.zeros((N_queries, dim_master), dtype=np.float32)
    features_master_dataset = np.zeros((N_dataset, dim_master), dtype=np.float32)

    # queries: get ROI-pooling features
    for i in tqdm(range(N_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_query_filename(i),
                                                                       roi=dataset.get_query_roi(i))
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward()

        pooled_rois_master_queries = np.squeeze(net.blobs['pooled_rois/normalized_flat'].data)
        features_master_rmac = pooled_rois_master_queries.sum(axis=0).reshape(-1, dim_master)
        features_master_rmac_norm = features_master_rmac / np.expand_dims(
            eps + np.sqrt((features_master_rmac ** 2).sum(axis=1)), axis=1)
        features_master_queries[i] = features_master_rmac_norm

    # dataset: get ROI-pooling features
    for i in tqdm(range(N_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
        # Load image, process image, get image regions, feed into the network, get descriptor, and store
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_filename(i), roi=None)

        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward()

        pooled_rois_master_dataset = np.squeeze(net.blobs['pooled_rois/normalized_flat'].data)
        features_master_rmac = pooled_rois_master_dataset.sum(axis=0).reshape(-1, dim_master)
        features_master_rmac_norm = features_master_rmac / np.expand_dims(
            eps + np.sqrt((features_master_rmac ** 2).sum(axis=1)), axis=1)
        features_master_dataset[i] = features_master_rmac_norm

    # Compute similarity
    sim = features_master_queries.dot(features_master_dataset.T)

    # Score
    dataset.score(sim, args.temp_dir, args.eval_binary)
