# -*- coding: utf-8 -*-

# usage: python ./myPython/raw_concat_test.py
#   --proto ./proto/branch_features_resnet101_normpython.prototxt
#   --weights ./caffemodel/deep_image_retrieval_model.caffemodel
#   --temp_dir ./eval/eval_test/

import os
import sys
import numpy as np
import caffe
import argparse
from tqdm import tqdm
from class_helper import *
from sklearn.decomposition import PCA


def get_rmac_features(I, R, net, end1=None, end2=None):
    net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
    net.blobs['data'].data[:] = I
    net.blobs['rois'].reshape(R.shape[0], R.shape[1])
    net.blobs['rois'].data[:] = R.astype(np.float32)
    net.forward()
    return np.squeeze(net.blobs[end1].data), \
           np.squeeze(net.blobs[end2].data)


def preprocessing_pooled_rois_feature(array_temp):
    shape = array_temp.shape
    vector_list = []
    for row in range(shape[0]):
        for col in range(shape[1]):
            if np.sum((array_temp[row, col, :]) ** 2) != 0.:
                vector_list.append(array_temp[row, col, :])
            else:
                break
    return np.array(vector_list)


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
    parser.add_argument('--features_dir', type=str, required=False,
                        help='Path to a temporary directory to store ROI-pooling features and PCA transformation')
    parser.set_defaults(gpu=0)
    parser.set_defaults(S=512)
    parser.set_defaults(L=2)
    parser.set_defaults(dataset_name='Oxford')
    parser.set_defaults(dataset='/home/processyuan/data/Oxford/')
    parser.set_defaults(eval_binary='/home/processyuan/NetworkOptimization/deep-retrieval/eval/compute_ap')
    parser.set_defaults(features_dir='/home/processyuan/NetworkOptimization/deep-retrieval/features/')
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

    # features are extracted here
    master_end_layer = 'pooled_rois/normalized_flat'
    branch_end_layer = 'pooled_rois_branch/normalized_flat'

    dim_master = net.blobs[master_end_layer].data.shape[1]
    dim_branch = net.blobs[branch_end_layer].data.shape[1]
    dim_features = dim_master + dim_branch

    N_queries = dataset.N_queries
    N_dataset = dataset.N_images
    N_REG_MAX = 23  # needs to improve later, size includes 8(many)/11/14/17/.../23(largest)

    features_queries = np.zeros((N_queries, dim_features), dtype=np.float32)
    features_dataset = np.zeros((N_dataset, dim_features), dtype=np.float32)
    features_master_queries = np.zeros((N_queries, dim_master), dtype=np.float32)
    features_master_dataset = np.zeros((N_dataset, dim_master), dtype=np.float32)
    features_branch_queries = np.zeros((N_queries, dim_branch), dtype=np.float32)
    features_branch_dataset = np.zeros((N_dataset, dim_branch), dtype=np.float32)
    pooled_rois_branch_queries_raw = np.zeros((N_queries, N_REG_MAX, dim_branch), dtype=np.float32)
    pooled_rois_branch_dataset_raw = np.zeros((N_dataset, N_REG_MAX, dim_branch), dtype=np.float32)
    pooled_rois_branch_queries_list = []
    pooled_rois_branch_dataset_list = []
    pooled_rois_master_queries_raw = np.zeros((N_queries, N_REG_MAX, dim_master), dtype=np.float32)
    pooled_rois_master_dataset_raw = np.zeros((N_dataset, N_REG_MAX, dim_master), dtype=np.float32)
    pooled_rois_master_queries_list = []
    pooled_rois_master_dataset_list = []

    # queries: get ROI-pooling features
    for i in tqdm(range(N_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_query_filename(i),
                                                                       roi=dataset.get_query_roi(i))
        pooled_rois_queries_master_temp, pooled_rois_queries_branch_temp = get_rmac_features(I, R, net,
                                                                                 master_end_layer, branch_end_layer)
        pooled_rois_branch_queries_raw[i, :R.shape[0], :] = pooled_rois_queries_branch_temp
        pooled_rois_master_queries_raw[i, :R.shape[0], :] = pooled_rois_queries_master_temp
        pooled_rois_branch_queries_list.append(pooled_rois_queries_branch_temp)
        pooled_rois_master_queries_list.append(pooled_rois_queries_master_temp)

    pooled_rois_branch_queries = preprocessing_pooled_rois_feature(pooled_rois_branch_queries_raw)
    pooled_rois_master_queries = preprocessing_pooled_rois_feature(pooled_rois_master_queries_raw)
    pooled_rois_branch_queries_fname = "{0}{1}_S{2}_L{3}_ROIpooling_branch_queries.npy".\
        format(args.features_dir, args.dataset_name, S, L)
    pooled_rois_master_queries_fname = "{0}{1}_S{2}_L{3}_ROIpooling_master_queries.npy". \
        format(args.features_dir, args.dataset_name, S, L)
    np.save(pooled_rois_branch_queries_fname, pooled_rois_branch_queries)
    np.save(pooled_rois_master_queries_fname, pooled_rois_master_queries)

    # dataset: get ROI-pooling features
    for i in tqdm(range(N_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
        # Load image, process image, get image regions, feed into the network, get descriptor, and store
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_filename(i), roi=None)
        pooled_rois_dataset_master_temp, pooled_rois_dataset_branch_temp = get_rmac_features(I, R, net,
                                                                                             master_end_layer,
                                                                                             branch_end_layer)
        pooled_rois_branch_dataset_raw[i, :R.shape[0], :] = pooled_rois_dataset_branch_temp
        pooled_rois_master_dataset_raw[i, :R.shape[0], :] = pooled_rois_dataset_master_temp
        pooled_rois_branch_dataset_list.append(pooled_rois_dataset_branch_temp)
        pooled_rois_master_dataset_list.append(pooled_rois_dataset_master_temp)

    pooled_rois_branch_dataset = preprocessing_pooled_rois_feature(pooled_rois_branch_dataset_raw)
    pooled_rois_master_dataset = preprocessing_pooled_rois_feature(pooled_rois_master_dataset_raw)
    pooled_rois_branch_dataset_fname = "{0}{1}_S{2}_L{3}_ROIpooling_branch_dataset.npy". \
        format(args.features_dir, args.dataset_name, S, L)
    pooled_rois_master_dataset_fname = "{0}{1}_S{2}_L{3}_ROIpooling_master_dataset.npy". \
        format(args.features_dir, args.dataset_name, S, L)
    np.save(pooled_rois_branch_dataset_fname, pooled_rois_branch_dataset)
    np.save(pooled_rois_master_dataset_fname, pooled_rois_master_dataset)

    pooled_rois_branch_features = np.vstack((pooled_rois_branch_queries, pooled_rois_branch_dataset))
    pooled_rois_master_features = np.vstack((pooled_rois_master_queries, pooled_rois_master_dataset))

    # perform PCA
    pca_branch = PCA(n_components=dim_branch, whiten=False)
    pca_branch.fit(pooled_rois_branch_features)
    np.save("{0}branch_ROIpooling_PCA_components.npy".format(args.features_dir), pca_branch.components_)
    np.save("{0}branch_ROIpooling_PCA_mean.npy".format(args.features_dir), pca_branch.mean_)
    np.save("{0}branch_ROIpooling_PCA_variance.npy".format(args.features_dir), pca_branch.explained_variance_)

    pca_master = PCA(n_components=dim_master, whiten=False)
    pca_master.fit(pooled_rois_master_features)
    np.save("{0}master_ROIpooling_PCA_components.npy".format(args.features_dir), pca_master.components_)
    np.save("{0}master_ROIpooling_PCA_mean.npy".format(args.features_dir), pca_master.mean_)
    np.save("{0}master_ROIpooling_PCA_variance.npy".format(args.features_dir), pca_master.explained_variance_)

    eps = 1e-8

    for i in tqdm(range(N_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
        features_branch_pca = pca_branch.transform(pooled_rois_branch_queries_list[i])
        features_branch_pca_norm = features_branch_pca / np.expand_dims(
            eps + np.sqrt((features_branch_pca ** 2).sum(axis=1)), axis=1)
        features_branch_rmac = features_branch_pca_norm.sum(axis=0).reshape(-1, dim_branch)
        features_branch_rmac_norm = features_branch_rmac / np.expand_dims(
            eps + np.sqrt((features_branch_rmac ** 2).sum(axis=1)), axis=1)
        features_branch_queries[i] = features_branch_rmac_norm

    for i in tqdm(range(N_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
        features_master_pca = pca_master.transform(pooled_rois_master_queries_list[i])
        features_master_pca_norm = features_master_pca / np.expand_dims(
            eps + np.sqrt((features_master_pca ** 2).sum(axis=1)), axis=1)
        features_master_rmac = features_master_pca_norm.sum(axis=0).reshape(-1, dim_master)
        features_master_rmac_norm = features_master_rmac / np.expand_dims(
            eps + np.sqrt((features_master_rmac ** 2).sum(axis=1)), axis=1)
        features_master_queries[i] = features_master_rmac_norm

    features_queries = np.hstack((features_branch_queries, features_master_queries))
    features_queries /= np.sqrt((features_queries * features_queries).sum(axis=1))[:, None]

    for i in tqdm(range(N_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
        features_branch_pca = pca_branch.transform(pooled_rois_branch_dataset_list[i])
        features_branch_pca_norm = features_branch_pca / np.expand_dims(
            eps + np.sqrt((features_branch_pca ** 2).sum(axis=1)), axis=1)
        features_branch_rmac = features_branch_pca_norm.sum(axis=0).reshape(-1, dim_branch)
        features_branch_rmac_norm = features_branch_rmac / np.expand_dims(
            eps + np.sqrt((features_branch_rmac ** 2).sum(axis=1)), axis=1)
        features_branch_dataset[i] = features_branch_rmac_norm

    for i in tqdm(range(N_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
        features_master_pca = pca_master.transform(pooled_rois_master_dataset_list[i])
        features_master_pca_norm = features_master_pca / np.expand_dims(
            eps + np.sqrt((features_master_pca ** 2).sum(axis=1)), axis=1)
        features_master_rmac = features_master_pca_norm.sum(axis=0).reshape(-1, dim_master)
        features_master_rmac_norm = features_master_rmac / np.expand_dims(
            eps + np.sqrt((features_master_rmac ** 2).sum(axis=1)), axis=1)
        features_master_dataset[i] = features_master_rmac_norm

    features_dataset = np.hstack((features_branch_dataset, features_master_dataset))
    features_dataset /= np.sqrt((features_dataset * features_dataset).sum(axis=1))[:, None]

    # Compute similarity
    sim = features_queries.dot(features_dataset.T)

    # Score
    dataset.score(sim, args.temp_dir, args.eval_binary)
