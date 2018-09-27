# -*- coding: utf-8 -*-

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

    # # import the model
    # model = caffe.proto.caffe_pb2.NetParameter()
    # f_caffemodel = open(args.model, 'rb')
    # model.ParseFromString(f_caffemodel.read())
    # f_caffemodel.close()

    # build the net
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # print the shape of weight blob stored in caffemodel
    for layer in net.params.keys():
        print(layer)
        for dim in range(len(net.params[layer])):
            print(net.params[layer][dim].data.shape)

    # # layers which follow ResNet-101 and have parameters in caffemodel
    # print net.params['pooled_rois/centered'][0].data
    # print net.params['pooled_rois/centered'][1].data
    # print net.params['pooled_rois/pca'][0].data
    # print net.params['pooled_rois/pca'][1].data
