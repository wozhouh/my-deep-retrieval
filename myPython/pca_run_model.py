# -*- coding: utf-8 -*-

# usage: python ./myPython/pca_run_model.py
#   --proto ./proto/branch_features_resnet101_normpython.prototxt
#   --weights ./caffemodel/deep_image_retrieval_model.caffemodel
#   --temp_dir ./eval/eval_test/
#   --pooled_rois_queries ./features/pca_concat/Oxford_S512_L2_ROIpooling_queries.npy
#   --pooled_rois_dataset ./features/pca_concat/Oxford_S512_L2_ROIpooling_dataset.npy

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
    parser.add_argument('--temp_dir', type=str, required=True,
                        help='Path to a temporary directory to store features and scores')
    parser.add_argument('--features_dir', type=str, required=False,
                        help='Path to a temporary directory to store ROI-pooling features and PCA transformation')
    parser.add_argument('--pooled_rois_queries', type=str, required=True,
                        help='Path to a store the ROI-pooling features for queries')
    parser.add_argument('--pooled_rois_dataset', type=str, required=True,
                        help='Path to a store the ROI-pooling features for dataset')
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

    # setting
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # Load the dataset and the image helper
    dataset = Dataset(args.dataset, args.eval_binary)
    image_helper = ImageHelper(S, L)
    N_queries = dataset.N_queries
    N_dataset = dataset.N_images
    eps = 1e-8

    # features are extracted here
    master_end_layer = 'rmac/normalized'
    branch_end_layer = 'pooled_rois_branch/normalized_flat'

    dim_master = net.blobs[master_end_layer].data.shape[1]
    dim_branch = net.blobs[branch_end_layer].data.shape[1]
    features_master_queries = np.zeros((N_queries, dim_master), dtype=np.float32)
    features_master_dataset = np.zeros((N_dataset, dim_master), dtype=np.float32)
    features_branch_queries = np.zeros((N_queries, dim_branch), dtype=np.float32)
    features_branch_dataset = np.zeros((N_dataset, dim_branch), dtype=np.float32)

    # load the data of features
    q, d = np.load(args.pooled_rois_queries), np.load(args.pooled_rois_dataset)
    pooled_rois_features = np.vstack((q, d))

    # perform PCA by scikit-learn
    scaler = preprocessing.StandardScaler().fit(pooled_rois_features)
    pca = PCA(n_components=pooled_rois_features.shape[1], copy=True, whiten=True)
    pca.fit(pooled_rois_features)
    np.save("{0}branch_ROIpooling_PCA_components.npy".format(args.features_dir), pca.components_)
    np.save("{0}branch_ROIpooling_PCA_mean.npy".format(args.features_dir), pca.mean_)
    np.save("{0}branch_ROIpooling_PCA_variance.npy".format(args.features_dir), pca.explained_variance_)

    for i in tqdm(range(N_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_query_filename(i),
                                                                       roi=dataset.get_query_roi(i))
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward()

        features_master_queries[i] = np.squeeze(net.blobs[master_end_layer].data)
        pooled_rois_queries_temp = np.squeeze(net.blobs[branch_end_layer].data)
        pooled_rois_queries_scale = scaler.transform(pooled_rois_queries_temp)
        features_branch_queries_pca = pca.transform(pooled_rois_queries_scale)
        features_branch_queries_pca_norm = features_branch_queries_pca / np.expand_dims(
            eps + np.sqrt((features_branch_queries_pca ** 2).sum(axis=1)), axis=1)
        features_branch_queries_rmac = features_branch_queries_pca_norm.sum(axis=0).reshape(1, -1)
        features_branch_queries_rmac_norm = features_branch_queries_rmac / \
                                            np.sqrt((features_branch_queries_rmac ** 2).sum(axis=1))[:, None]
        features_branch_queries[i] = features_branch_queries_rmac_norm

    features_branch_queries_fname = "{0}features_branch_queries.npy".format(args.features_dir)
    features_master_queries_fname = "{0}features_master_queries.npy".format(args.features_dir)
    np.save(features_branch_queries_fname, features_branch_queries)
    np.save(features_master_queries_fname, features_master_queries)
    features_queries = np.hstack((features_branch_queries, features_master_queries))
    features_queries /= np.sqrt((features_queries * features_queries).sum(axis=1))[:, None]

    for i in tqdm(range(N_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_filename(i), roi=None)
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward()

        features_master_dataset[i] = np.squeeze(net.blobs[master_end_layer].data)
        pooled_rois_dataset_temp = np.squeeze(net.blobs[branch_end_layer].data)
        pooled_rois_dataset_scale = scaler.transform(pooled_rois_dataset_temp)
        features_branch_dataset_pca = pca.transform(pooled_rois_dataset_scale)
        features_branch_dataset_pca_norm = features_branch_dataset_pca / np.expand_dims(
            eps + np.sqrt((features_branch_dataset_pca ** 2).sum(axis=1)), axis=1)
        features_branch_dataset_rmac = features_branch_dataset_pca_norm.sum(axis=0).reshape(1, -1)
        features_branch_dataset_rmac_norm = features_branch_dataset_rmac / \
                                            np.sqrt((features_branch_dataset_rmac ** 2).sum(axis=1))[:, None]
        features_branch_dataset[i] = features_branch_dataset_rmac_norm

    features_branch_dataset_fname = "{0}features_branch_dataset.npy".format(args.features_dir)
    features_master_dataset_fname = "{0}features_master_dataset.npy".format(args.features_dir)
    np.save(features_branch_dataset_fname, features_branch_dataset)
    np.save(features_master_dataset_fname, features_master_dataset)
    features_dataset = np.hstack((features_branch_dataset, features_master_dataset))
    features_dataset /= np.sqrt((features_dataset * features_dataset).sum(axis=1))[:, None]

    # Compute similarity
    sim = features_queries.dot(features_dataset.T)

    # Score
    dataset.score(sim, args.temp_dir, args.eval_binary)
