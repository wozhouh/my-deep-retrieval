# -*- coding: utf-8 -*-

# usage: python ./myPython/rmac_eltwise_get_features.py
#   --proto ./proto/rmac_eltwise_resnet101_normpython.prototxt
#   --weights ./caffemodel/deep_image_retrieval_model.caffemodel

import sys
import numpy as np
import caffe
import argparse
from tqdm import tqdm
from class_helper import *

if __name__ == '__main__':

    # Config
    parser = argparse.ArgumentParser(description='PCA features preprocessing')
    parser.add_argument('--gpu', type=int, required=False, help='GPU ID to use (e.g. 0)')
    parser.add_argument('--S', type=int, required=False, help='Resize larger side of image to S pixels (e.g. 800)')
    parser.add_argument('--L', type=int, required=False, help='Use L spatial levels (e.g. 2)')
    parser.add_argument('--proto', type=str, required=True, help='Path to the prototxt file')
    parser.add_argument('--weights', type=str, required=True, help='Path to the caffemodel file')
    parser.add_argument('--dataset', type=str, required=False, help='Path to the Oxford / Paris directory')
    parser.add_argument('--dataset_name', type=str, required=False, help='Dataset name')
    parser.add_argument('--eval_binary', type=str, required=False,
                        help='Path to the compute_ap binary to evaluate Oxford / Paris')
    parser.add_argument('--features_dir', type=str, required=False,
                        help='Path to a temporary directory to store ROI-pooling features and PCA transformation')
    parser.set_defaults(gpu=0)
    parser.set_defaults(S=512)
    parser.set_defaults(L=2)
    parser.set_defaults(dataset_name='Oxford')
    parser.set_defaults(dataset='/home/processyuan/data/Oxford/')
    parser.set_defaults(eval_binary='/home/processyuan/NetworkOptimization/deep-retrieval/eval/compute_ap')
    parser.set_defaults(features_dir='/home/processyuan/NetworkOptimization/deep-retrieval/features/rmac_eltwise/')
    args = parser.parse_args()

    S = args.S
    L = args.L

    # Configure caffe and load the network
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # Load the dataset and the image helper
    dataset = Dataset(args.dataset)
    image_helper = ImageHelper(S, L)

    # Features are extracted here
    master = 'rmac/normalized'
    branch = ['rmac_branch_32/normalized',
              'rmac_branch_16/normalized',
              'rmac_branch_8/normalized',
              'rmac_branch_4/normalized']
    num_branch = len(branch)

    N_queries = dataset.N_queries
    N_dataset = dataset.N_images
    dim_branch = [net.blobs[branch[k]].data.shape[1] for k in range(num_branch)]
    dim_features = np.sum(dim_branch)
    rmac_queries_list = [np.zeros((N_queries, dim_branch[k]), dtype=np.float32) for k in range(num_branch)]
    rmac_dataset_list = [np.zeros((N_dataset, dim_branch[k]), dtype=np.float32) for k in range(num_branch)]

    # queries: get concat RMAC features
    for i in tqdm(range(N_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_query_filename(i),
                                                                       roi=dataset.get_query_roi(i))
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end=master)
        for k in range(num_branch):
            (rmac_queries_list[k])[i] = np.squeeze(net.blobs[branch[k]].data) * dim_branch[k]

    rmac_queries = np.hstack((rmac_queries_list[k] for k in range(num_branch)))
    rmac_queries_fname = "{0}{1}_S{2}_L{3}_rmac_queries.npy"\
        .format(args.features_dir, args.dataset_name, S, L)
    np.save(rmac_queries_fname, rmac_queries)

    # dataset: get concat RMAC features
    for i in tqdm(range(N_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
        # Load image, process image, get image regions, feed into the network, get descriptor, and store
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_filename(i), roi=None)
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end=master)
        for k in range(num_branch):
            (rmac_dataset_list[k])[i] = np.squeeze(net.blobs[branch[k]].data) * dim_branch[k]

    rmac_dataset = np.hstack((rmac_dataset_list[k] for k in range(num_branch)))
    rmac_dataset_fname = "{0}{1}_S{2}_L{3}_rmac_dataset.npy" \
        .format(args.features_dir, args.dataset_name, S, L)
    np.save(rmac_dataset_fname, rmac_dataset)
