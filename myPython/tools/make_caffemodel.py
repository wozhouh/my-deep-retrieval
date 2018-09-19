# -*- coding: utf-8 -*-

# Python script for revision of weight blob in .caffemodel manually

# when adds weight of new layer to caffemodel, need to build a net by Python first then assign values
# (also a corresponding prototxt is needed)

import caffe
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn import preprocessing

if __name__ == "__main__":
    # configure
    parser = argparse.ArgumentParser(description='revise a caffemodel for training')
    parser.add_argument('--proto', type=str, required=True, help='Path to the prototxt file for a network with branch '
                                                                 'concatenation')
    parser.add_argument('--weights', type=str, required=True, help='Path to the original caffemodel file')
    parser.add_argument('--features', type=str, required=True, help='Path to the extracted features vectors '
                                                                         'on the whole dataset')
    args = parser.parse_args()
    caffemodel_out = args.weights[0:-11] + '_init.caffemodel'

    # setting
    caffe.set_mode_cpu()

    # build the net
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # pre-process the features by scaling and PCA, which used to initialized the weight later
    features = np.load(args.features)
    scaler = preprocessing.StandardScaler().fit(features)
    mean_value = scaler.mean_
    # var_value = scaler.scale_
    features_scale = scaler.transform(features)
    pca = PCA(n_components=features.shape[1], copy=True, whiten=True)
    pca.fit(features_scale)
    ip_weight = pca.components_
    # ip_weight = pca.components_ / (var_value.reshape(1, -1))

    # copy the weight from master to branch
    net.params['pooled_rois_branch_1/pca'][0].data[...] = ip_weight
    net.params['pooled_rois_branch_1/centered'][1].data[...] = -mean_value

    # print for checking
    layers = ['pooled_rois_branch_1/pca', 'pooled_rois_branch_1/centered']
    for l in layers:
        for k in range(len(net.params[l])):
            print(net.params[l][k].data)

    # save the model
    net.save(caffemodel_out)
