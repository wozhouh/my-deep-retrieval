# -*- coding: utf-8 -*-

# Run after pca_get_features.py

# usage: python ./myPython/rmac_eltwise_retrieval.py
#   --proto ./proto/rmac_eltwise_resnet101_normpython.prototxt
#   --weights ./caffemodel/deep_image_retrieval_model.caffemodel

import sys
import numpy as np
import caffe
import argparse
from tqdm import tqdm
from class_helper import *
from sklearn.decomposition import PCA
from sklearn import preprocessing

if __name__ == '__main__':

    # Config
    parser = argparse.ArgumentParser(description='Evaluate Oxford')
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
    parser.set_defaults(features_dir='/home/processyuan/NetworkOptimization/deep-retrieval/features/rmac_eltwise/')
    args = parser.parse_args()

    S = args.S
    L = args.L

    # setting
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # Load the dataset and the image helper
    dataset = Dataset(args.dataset, args.eval_binary)
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
    dim_master = net.blobs[master].data.shape[1]
    eps = 1e-8

    # saved region-wise features
    rmac_queries_fname = "{0}{1}_S{2}_L{3}_rmac_queries.npy" \
        .format(args.features_dir, args.dataset_name, S, L)
    rmac_dataset_fname = "{0}{1}_S{2}_L{3}_rmac_dataset.npy" \
        .format(args.features_dir, args.dataset_name, S, L)

    features_master_queries = np.zeros((N_queries, dim_master), dtype=np.float32)
    features_master_dataset = np.zeros((N_dataset, dim_master), dtype=np.float32)
    features_branch_queries = np.zeros((N_queries, dim_master), dtype=np.float32)
    features_branch_dataset = np.zeros((N_dataset, dim_master), dtype=np.float32)
    rmac_queries_list = [np.zeros((N_queries, dim_branch[k]), dtype=np.float32) for k in range(num_branch)]
    rmac_dataset_list = [np.zeros((N_dataset, dim_branch[k]), dtype=np.float32) for k in range(num_branch)]

    # load the data of features and perform PCA by scikit-learn
    q, d = np.load(rmac_queries_fname), np.load(rmac_dataset_fname)
    rmac_features = np.vstack((q, d))
    scaler = preprocessing.StandardScaler().fit(rmac_features)
    rmac_features_scaler = scaler.transform(rmac_features)
    pca = PCA(n_components=dim_master, copy=True, whiten=True)
    pca.fit(rmac_features_scaler)
    np.save("{0}rmac_PCA_components.npy".format(args.features_dir), pca.components_)
    np.save("{0}rmac_PCA_mean.npy".format(args.features_dir), pca.mean_)
    np.save("{0}rmac_PCA_variance.npy".format(args.features_dir), pca.explained_variance_)

    # First part, queries
    for i in tqdm(range(N_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_query_filename(i),
                                                                       roi=dataset.get_query_roi(i))
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end=master)

        features_master_queries[i] = np.squeeze(net.blobs[master].data)
        for k in range(num_branch):
            (rmac_queries_list[k])[i] = np.squeeze(net.blobs[branch[k]].data)
        features_branch_queries_temp = np.hstack(((rmac_queries_list[k])[i] for k in range(num_branch)))
        features_branch_queries_scaler = scaler.transform(features_branch_queries_temp.reshape(1, -1))
        features_branch_queries_pca = pca.transform(features_branch_queries_scaler)
        features_branch_queries[i] = features_branch_queries_pca / np.expand_dims(
            eps + np.sqrt((features_branch_queries_pca ** 2).sum(axis=1)), axis=1)

    features_queries = np.dstack((features_master_queries, features_branch_queries)).sum(axis=2)
    features_queries /= np.sqrt((features_queries ** 2).sum(axis=1))[:, None]

    # Second part, dataset
    for i in tqdm(range(N_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_filename(i), roi=None)
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end=master)

        features_master_dataset[i] = np.squeeze(net.blobs[master].data)
        for k in range(num_branch):
            (rmac_dataset_list[k])[i] = np.squeeze(net.blobs[branch[k]].data)
        features_branch_dataset_temp = np.hstack(((rmac_dataset_list[k])[i] for k in range(num_branch)))
        features_branch_dataset_scaler = scaler.transform(features_branch_dataset_temp.reshape(1, -1))
        features_branch_dataset_pca = pca.transform(features_branch_dataset_scaler)
        features_branch_dataset[i] = features_branch_dataset_pca / np.expand_dims(
            eps + np.sqrt((features_branch_dataset_pca ** 2).sum(axis=1)), axis=1)

    features_dataset = np.dstack((features_master_dataset, features_branch_dataset)).sum(axis=2)
    features_dataset /= np.sqrt((features_dataset ** 2).sum(axis=1))[:, None]

    # Compute similarity
    sim = features_queries.dot(features_dataset.T)

    # Score
    dataset.score(sim, args.temp_dir, args.eval_binary)
