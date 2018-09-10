# -*- coding: utf-8 -*-

# usage: python ./myPython/raw_concat_test.py
#   --proto ./proto/branch_features_resnet101_normpython.prototxt
#   --weights ./caffemodel/deep_image_retrieval_model.caffemodel
#   --dataset_features
#   --queries_features

import numpy as np
import caffe
import argparse
from sklearn.decomposition import PCA


if __name__ == '__main__':

    # configure
    parser = argparse.ArgumentParser(description='perform PCA and store the transformation matrix into caffemodel')
    parser.add_argument('--proto', type=str, required=True, help='Path to the prototxt file for a network with PCA '
                                                                 'concatenation')
    parser.add_argument('--weights', type=str, required=True, help='Path to the caffemodel file')
    parser.add_argument('--features_master_queries', type=str, required=True, help='stores master features of queries')
    parser.add_argument('--queries_features', type=str, required=True, help='feature vector after ROI-pooling '
                                                                            'extracted from the queries')

    args = parser.parse_args()

    # setting
    caffe.set_mode_cpu()

    # load the data of features
    q, d = get_feature_vector(queries_npy=args.queries_features, dataset_npy=args.dataset_features)
    features = np.vstack((q, d))

    # perform PCA by scikit-learn
    pca = PCA(n_components=features.shape[1], whiten=True)
    pca.fit(features)
    fc_weight = pca.components_
    scale_bias = pca.mean_
