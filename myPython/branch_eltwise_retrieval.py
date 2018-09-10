# -*- coding: utf-8 -*-

# usage: python ./myPython/raw_concat_test.py
#   --proto ./proto/branch_concat_resnet101_normpython.prototxt
#   --weights ./caffemodel/deep_image_retrieval_model_branch_concat.caffemodel
#   --temp_dir ./eval/eval_test/

import os
import sys
import numpy as np
import caffe
import argparse
from tqdm import tqdm
from class_helper import *


def get_rmac_features(I, R, net):
    net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
    net.blobs['data'].data[:] = I
    net.blobs['rois'].reshape(R.shape[0], R.shape[1])
    net.blobs['rois'].data[:] = R.astype(np.float32)
    net.forward()
    return np.squeeze(net.blobs['rmac/normalized'].data), \
           np.squeeze(net.blobs['rmac_branch/normalized'].data)


def extract_features(dataset, image_helper, net, args):
    S = args.S
    L = args.L
    out_master_queries_fname = "{0}/{1}_S{2}_L{3}_queries_master.npy".format(args.temp_dir, args.dataset_name, S, L)
    out_branch_queries_fname = "{0}/{1}_S{2}_L{3}_queries_branch.npy".format(args.temp_dir, args.dataset_name, S, L)
    dim_features = net.blobs['rmac/normalized'].data.shape[1]
    assert dim_features == net.blobs['rmac_branch/normalized'].data.shape[1]
    N_queries = dataset.N_queries
    features_master_queries = np.zeros((N_queries, dim_features), dtype=np.float32)
    features_branch_queries = np.zeros((N_queries, dim_features), dtype=np.float32)
    for i in tqdm(range(N_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
        # Load image, process image, get image regions, feed into the network, get descriptor, and store
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_query_filename(i),
                                                                       roi=dataset.get_query_roi(i))
        features_master_queries[i], features_branch_queries[i] = get_rmac_features(I, R, net)

    np.save(out_master_queries_fname, features_master_queries)
    np.save(out_branch_queries_fname, features_branch_queries)

    features_queries = np.dstack([features_master_queries, features_branch_queries]).sum(axis=2)
    features_queries /= np.sqrt((features_queries * features_queries).sum(axis=1))[:, None]

    # Second part, dataset
    out_master_dataset_fname = "{0}/{1}_S{2}_L{3}_dataset_master.npy".format(args.temp_dir, args.dataset_name, S, L)
    out_branch_dataset_fname = "{0}/{1}_S{2}_L{3}_dataset_branch.npy".format(args.temp_dir, args.dataset_name, S, L)
    N_dataset = dataset.N_images
    features_master_dataset = np.zeros((N_dataset, dim_features), dtype=np.float32)
    features_branch_dataset = np.zeros((N_dataset, dim_features), dtype=np.float32)
    for i in tqdm(range(N_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
        # Load image, process image, get image regions, feed into the network, get descriptor, and store
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_filename(i), roi=None)
        features_master_dataset[i], features_branch_dataset[i] = get_rmac_features(I, R, net)

    np.save(out_master_dataset_fname, features_master_dataset)
    np.save(out_branch_dataset_fname, features_branch_dataset)

    features_dataset = np.dstack([features_master_dataset, features_branch_dataset]).sum(axis=2)
    features_dataset /= np.sqrt((features_dataset * features_dataset).sum(axis=1))[:, None]

    return features_queries, features_dataset


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
    parser.add_argument('--temp_dir', type=str, required=True,
                        help='Path to a temporary directory to store features and scores')
    parser.set_defaults(gpu=0)
    parser.set_defaults(S=512)
    parser.set_defaults(L=2)
    parser.set_defaults(dataset_name='Oxford')
    parser.set_defaults(dataset='/home/processyuan/data/Oxford/')
    parser.set_defaults(eval_binary='/home/processyuan/NetworkOptimization/deep-retrieval/eval/compute_ap')
    args = parser.parse_args()

    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    # Configure caffe and load the network
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # Load the dataset and the image helper
    dataset = Dataset(args.dataset, args.eval_binary)
    image_helper = ImageHelper(args.S, args.L)

    # Extract features
    features_queries, features_dataset = extract_features(dataset, image_helper, net, args)

    # Compute similarity
    sim = features_queries.dot(features_dataset.T)

    # Score
    dataset.score(sim, args.temp_dir, args.eval_binary)
