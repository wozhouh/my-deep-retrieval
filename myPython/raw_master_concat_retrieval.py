# -*- coding: utf-8 -*-

# usage: python ./myPython/raw_concat_test.py
#   --proto ./proto/branch_features_resnet101_normpython.prototxt
#   --weights ./caffemodel/deep_image_retrieval_model.caffemodel

import os
import sys
import numpy as np
import caffe
import argparse
from tqdm import tqdm
from class_helper import *

if __name__ == '__main__':

    # Configure
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
    parser.add_argument('--features_dir', type=str, required=False,
                        help='Path to a temporary directory to store ROI-pooling features and PCA transformation')
    parser.set_defaults(gpu=0)
    parser.set_defaults(S=512)
    parser.set_defaults(L=2)
    parser.set_defaults(dataset_name='Oxford')
    parser.set_defaults(dataset='/home/processyuan/data/Oxford/')
    parser.set_defaults(eval_binary='/home/processyuan/NetworkOptimization/deep-retrieval/eval/compute_ap')
    parser.set_defaults(temp_dir='/home/processyuan/NetworkOptimization/deep-retrieval/eval/eval_test/')
    parser.set_defaults(features_dir='/home/processyuan/NetworkOptimization/deep-retrieval/features/raw_master_concat/')
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
    master = 'rmac/normalized'
    branch = ['rmac_branch_16/normalized',
              'rmac_branch_8/normalized',
              'rmac_branch_4/normalized']
    dim_master = net.blobs[master].data.shape[1]
    dim_branch = [net.blobs[branch[k]].data.shape[1] for k in range(len(branch))]
    num_branch = len(branch)
    eps = 1e-8

    features_queries_master = np.zeros((N_queries, dim_master), dtype=np.float32)
    features_dataset_master = np.zeros((N_dataset, dim_master), dtype=np.float32)
    features_queries_list = [np.zeros((N_queries, dim_branch[k]), dtype=np.float32) for k in range(num_branch)]
    features_dataset_list = [np.zeros((N_dataset, dim_branch[k]), dtype=np.float32) for k in range(num_branch)]

    # queries: get ROI-pooling features
    for i in tqdm(range(N_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_query_filename(i),
                                                                       roi=dataset.get_query_roi(i))
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end=master)
        features_queries_master[i] = np.squeeze(net.blobs[master].data) * dim_master
        for k in range(num_branch):
            (features_queries_list[k])[i] = np.squeeze(net.blobs[branch[k]].data) * dim_branch[k]

    features_queries_list.append(features_queries_master)
    features_queries = np.hstack((features_queries_list[k] for k in range(num_branch + 1)))
    features_queries /= np.sqrt((features_queries * features_queries).sum(axis=1))[:, None]

    # dataset: get ROI-pooling features
    for i in tqdm(range(N_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
        # Load image, process image, get image regions, feed into the network, get descriptor, and store
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_filename(i), roi=None)
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end=master)
        features_dataset_master[i] = np.squeeze(net.blobs[master].data) * dim_master
        for k in range(num_branch):
            (features_dataset_list[k])[i] = np.squeeze(net.blobs[branch[k]].data) * dim_branch[k]

    # Save the ROI-pooled features from middle layers
    features_queries_master_fname = "{0}{1}_S{2}_L{3}_master_queries.npy"\
        .format(args.features_dir, args.dataset_name, S, L)
    features_dataset_master_fname = "{0}{1}_S{2}_L{3}_master_dataset.npy"\
        .format(args.features_dir, args.dataset_name, S, L)
    np.save(features_queries_master_fname, features_queries_master)
    np.save(features_dataset_master_fname, features_dataset_master)

    features_dataset_list.append(features_dataset_master)

    # Concatenation and normalization

    features_dataset = np.hstack((features_dataset_list[k] for k in range(num_branch+1)))
    features_dataset /= np.sqrt((features_dataset * features_dataset).sum(axis=1))[:, None]

    # Compute similarity
    sim = features_queries.dot(features_dataset.T)

    # Score
    dataset.score(sim, args.temp_dir, args.eval_binary)
