# -*- coding: utf-8 -*-

# Python script for printing the shape of weight blob in the caffemodel

# usage: python ./myPython/check_caffemodel.py
#   --proto ./proto/branch_features_resnet101_normpython.prototxt
#   --weights ./caffemodel/deep_image_retrieval_model.caffemodel

import caffe
import argparse

if __name__ == "__main__":

    # configure
    parser = argparse.ArgumentParser(description='print the shape of weights stored in caffemodel')
    parser.add_argument('--proto', type=str, required=True, help='Path to the prototxt file')
    parser.add_argument('--weights', type=str, required=True, help='Path to the caffemodel file')
    args = parser.parse_args()

    # setting
    caffe.set_mode_cpu()

    # build the net
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # print the shape of weight blob stored in caffemodel
    for layer in net.params.keys():
        print(layer)
        for dim in range(len(net.params[layer])):
            print(net.params[layer][dim].data.shape)

    # print the weights for checking
    layers = ['pooled_rois/centered', 'pooled_rois_branch_1/centered']
    for l in layers:
        for k in range(len(net.params[l])):
            print(net.params[l][k].data)
        print('\n')
