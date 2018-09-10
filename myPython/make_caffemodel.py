 # -*- coding: utf-8 -*-

import caffe
import argparse

if __name__ == "__main__":

    # configure
    parser = argparse.ArgumentParser(description='make a caffemodel adapted for branch concatenation')
    parser.add_argument('--proto', type=str, required=True, help='Path to the prototxt file for a network with branch '
                                                                 'concatenation')
    parser.add_argument('--weights', type=str, required=True, help='Path to the original caffemodel file')
    args = parser.parse_args()
    caffemodel_out = args.model[0:-11] + '_branch_eltwise.caffemodel'

    # setting
    caffe.set_mode_cpu()

    # build the net
    net = caffe.Net(args.proto, args.model, caffe.TEST)

    # copy the weight from master to branch
    net.params['pooled_rois_branch/pca'][0].data[...] = net.params['pooled_rois/pca'][0].data[...]
    net.params['pooled_rois_branch/pca'][1].data[...] = net.params['pooled_rois/pca'][1].data[...]
    net.params['pooled_rois_branch/centered'][0].data[...] = net.params['pooled_rois/centered'][0].data[...]
    net.params['pooled_rois_branch/centered'][1].data[...] = net.params['pooled_rois/centered'][1].data[...]

    # print for checking
    print net.params['pooled_rois_branch/pca'][0].data
    print net.params['pooled_rois/pca'][0].data
    print net.params['pooled_rois_branch/pca'][1].data
    print net.params['pooled_rois/pca'][1].data
    print net.params['pooled_rois_branch/centered'][0].data
    print net.params['pooled_rois/centered'][0].data
    print net.params['pooled_rois_branch/centered'][1].data
    print net.params['pooled_rois/centered'][1].data

    # save the model
    net.save(caffemodel_out)
