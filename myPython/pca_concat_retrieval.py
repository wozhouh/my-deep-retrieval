# -*- coding: utf-8 -*-

# Run after pca_get_features.py

# usage: python ./myPython/pca_concat_retrieval.py
#   --proto ./proto/pca_concat_resnet101_normpython.prototxt
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

    # features are extracted here, keep the same with pca_get_features.py
    master = 'rmac/normalized'
    branch = ['pooled_rois_branch_16/normalized',
              'pooled_rois_branch_8/normalized',
              'pooled_rois_branch_4/normalized']

    num_branch = len(branch)
    dim_master = net.blobs[master].data.shape[1]
    dim_branch = [net.blobs[branch[k]].data.shape[1] for k in range(len(branch))]
    features_master_queries = np.zeros((N_queries, dim_master), dtype=np.float32)
    features_master_dataset = np.zeros((N_dataset, dim_master), dtype=np.float32)
    features_queries_list = [np.zeros((N_queries, dim_branch[k]), dtype=np.float32) for k in range(num_branch)]
    features_dataset_list = [np.zeros((N_dataset, dim_branch[k]), dtype=np.float32) for k in range(num_branch)]
    pca = []
    scaler = []

    # saved region-wise features
    pooled_rois_queries_fname = ["{0}{1}_S{2}_L{3}_ROIpooling_branch{4}_queries.npy".
                                     format(args.features_dir, args.dataset_name, S, L, k) for k in range(num_branch)]
    pooled_rois_dataset_fname = ["{0}{1}_S{2}_L{3}_ROIpooling_branch{4}_dataset.npy".
                                     format(args.features_dir, args.dataset_name, S, L, k) for k in range(num_branch)]

    # load the data of features and perform PCA by scikit-learn
    for k in range(num_branch):
        q, d = np.load(pooled_rois_queries_fname[k]), np.load(pooled_rois_dataset_fname[k])
        pooled_rois_features = np.vstack((q, d))
        scaler.append(preprocessing.StandardScaler().fit(pooled_rois_features))
        pca_temp = PCA(n_components=pooled_rois_features.shape[1], copy=True, whiten=True)
        pca_temp.fit(pooled_rois_features)
        np.save("{0}branch{1}_ROIpooling_PCA_components.npy".format(args.features_dir, k), pca_temp.components_)
        np.save("{0}branch{1}_ROIpooling_PCA_mean.npy".format(args.features_dir, k), pca_temp.mean_)
        np.save("{0}branch{1}_ROIpooling_PCA_variance.npy".format(args.features_dir, k), pca_temp.explained_variance_)
        pca.append(pca_temp)

    # First part, queries
    for i in tqdm(range(N_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_query_filename(i),
                                                                       roi=dataset.get_query_roi(i))
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end=master)

        features_master_queries[i] = np.squeeze(net.blobs[master].data) * dim_master
        pooled_rois_queries_temp = [np.squeeze(net.blobs[branch[k]].data) for k in range(num_branch)]
        pooled_rois_queries_scale = [scaler[k].transform(pooled_rois_queries_temp[k]) for k in range(num_branch)]
        features_branch_queries_pca = [pca[k].transform(pooled_rois_queries_scale[k]) for k in range(num_branch)]
        features_branch_queries_pca_norm = [features_branch_queries_pca[k] / np.expand_dims(
            eps + np.sqrt((features_branch_queries_pca[k] ** 2).sum(axis=1)), axis=1)
                                            for k in range(num_branch)]
        features_branch_queries_rmac = [features_branch_queries_pca_norm[k].sum(axis=0).reshape(1, -1)
                                        for k in range(num_branch)]

        features_branch_queries_rmac_norm = [features_branch_queries_rmac[k] /
                                            np.sqrt((features_branch_queries_rmac[k] ** 2).sum(axis=1))[:, None]
                                             for k in range(num_branch)]
        for k in range(num_branch):
            (features_queries_list[k])[i] = features_branch_queries_rmac_norm[k] * dim_branch[k]

    features_queries_list.append(features_master_queries)
    features_queries = np.hstack((features_queries_list[k] for k in range(num_branch + 1)))
    features_queries /= np.sqrt((features_queries * features_queries).sum(axis=1))[:, None]

    # Second part, dataset
    for i in tqdm(range(N_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
        I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_filename(i), roi=None)
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end=master)

        features_master_dataset[i] = np.squeeze(net.blobs[master].data) * dim_master
        pooled_rois_dataset_temp = [np.squeeze(net.blobs[branch[k]].data) for k in range(num_branch)]
        pooled_rois_dataset_scale = [scaler[k].transform(pooled_rois_dataset_temp[k]) for k in range(num_branch)]
        features_branch_dataset_pca = [pca[k].transform(pooled_rois_dataset_scale[k]) for k in range(num_branch)]
        features_branch_dataset_pca_norm = [features_branch_dataset_pca[k] / np.expand_dims(
            eps + np.sqrt((features_branch_dataset_pca[k] ** 2).sum(axis=1)), axis=1)
                                            for k in range(num_branch)]
        features_branch_dataset_rmac = [features_branch_dataset_pca_norm[k].sum(axis=0).reshape(1, -1)
                                        for k in range(num_branch)]

        features_branch_dataset_rmac_norm = [features_branch_dataset_rmac[k] /
                                             np.sqrt((features_branch_dataset_rmac[k] ** 2).sum(axis=1))[:, None]
                                             for k in range(num_branch)]
        for k in range(num_branch):
            (features_dataset_list[k])[i] = features_branch_dataset_rmac_norm[k] * dim_branch[k]

    features_dataset_list.append(features_master_dataset)
    features_dataset = np.hstack((features_dataset_list[k] for k in range(num_branch + 1)))
    features_dataset /= np.sqrt((features_dataset * features_dataset).sum(axis=1))[:, None]

    # Compute similarity
    sim = features_queries.dot(features_dataset.T)

    # Score
    dataset.score(sim, args.temp_dir, args.eval_binary)
