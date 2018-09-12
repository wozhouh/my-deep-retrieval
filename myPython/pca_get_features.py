# -*- coding: utf-8 -*-

# usage: python ./myPython/pca_get_features.py
#   --proto ./proto/branch_features_resnet101_normpython.prototxt
#   --weights ./caffemodel/deep_image_retrieval_model.caffemodel
#   --features_dir ./features/pca_concat/

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
    parser.set_defaults(features_dir='/home/processyuan/NetworkOptimization/deep-retrieval/features/pca_concat/')
    args = parser.parse_args()

    S = args.S
    L = args.L

    # Configure caffe and load the network
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # Load the dataset and the image helper
    dataset = Dataset(args.dataset, args.eval_binary)
    image_helper = ImageHelper(S, L)

    # Features are extracted here
    branch = ['pooled_rois_branch_16/normalized',
              'pooled_rois_branch_8/normalized',
              'pooled_rois_branch_4/normalized']
    num_branch = len(branch)

    N_queries = dataset.N_queries
    N_dataset = dataset.N_images
    pooled_rois_queries_list = []
    pooled_rois_dataset_list = []
    for k in range(num_branch):
        pooled_rois_queries_list.append([])
        pooled_rois_dataset_list.append([])

    # queries: get ROI-pooling features
    for i in tqdm(range(N_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_query_filename(i),
                                                                       roi=dataset.get_query_roi(i))
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end=branch[0])

        for k in range(num_branch):
            pooled_rois_queries_temp = np.squeeze(net.blobs[branch[k]].data)
            for l in range(pooled_rois_queries_temp.shape[0]):
                (pooled_rois_queries_list[k]).append(pooled_rois_queries_temp[l, :])

    pooled_rois_queries = [np.array(pooled_rois_queries_list[k]) for k in range(num_branch)]
    pooled_rois_queries_fname = ["{0}{1}_S{2}_L{3}_ROIpooling_branch{4}_queries.npy".
                                     format(args.features_dir, args.dataset_name, S, L, k) for k in range(num_branch)]
    for k in range(num_branch):
        np.save(pooled_rois_queries_fname[k], pooled_rois_queries[k])

    # dataset: get ROI-pooling features
    for i in tqdm(range(N_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
        # Load image, process image, get image regions, feed into the network, get descriptor, and store
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_filename(i), roi=None)
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end=branch[0])

        for k in range(num_branch):
            pooled_rois_dataset_temp = np.squeeze(net.blobs[branch[k]].data)
            for l in range(pooled_rois_dataset_temp.shape[0]):
                (pooled_rois_dataset_list[k]).append(pooled_rois_dataset_temp[l, :])

    pooled_rois_dataset = [np.array(pooled_rois_dataset_list[k]) for k in range(num_branch)]
    pooled_rois_dataset_fname = ["{0}{1}_S{2}_L{3}_ROIpooling_branch{4}_dataset.npy".
                                     format(args.features_dir, args.dataset_name, S, L, k) for k in range(num_branch)]
    for k in range(num_branch):
        np.save(pooled_rois_dataset_fname[k], pooled_rois_dataset[k])
