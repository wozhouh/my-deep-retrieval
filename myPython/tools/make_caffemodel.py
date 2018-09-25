# -*- coding: utf-8 -*-

# Python script for revision of weight blob in .caffemodel manually

# when adds weight of new layer to caffemodel, need to build a net by Python first then assign values
# (also a corresponding prototxt is needed)

# usage: python ./tools/make_caffemodel.py
#   --old_proto ../proto/deploy_resnet101.prototxt
#   --new_proto ../proto/train-distilling/resnet101_TeacherStudent.prototxt
#   --weights ../caffemodel/deep_image_retrieval_model.caffemodel

import caffe
import argparse


if __name__ == "__main__":
    # configure
    parser = argparse.ArgumentParser(description='revise a caffemodel for training')
    parser.add_argument('--old_proto', type=str, required=True, help='Path to the old prototxt file')
    parser.add_argument('--new_proto', type=str, required=True, help='Path to the new prototxt file')
    parser.add_argument('--weights', type=str, required=True, help='Path to the original caffemodel file')

    args = parser.parse_args()
    caffemodel_out = args.weights[0:-11] + '_distilling.caffemodel'

    # setting
    caffe.set_mode_cpu()

    # build the net
    net_in = caffe.Net(args.old_proto, args.weights, caffe.TEST)
    net_out = caffe.Net(args.new_proto, args.weights, caffe.TEST)

    # check_model = True
    # for k in net_in.params.keys():
    #     if not net_out.params.has_key(k):
    #         check_model = False
    #         print(k)
    # print(check_model)

    for l in net_in.params.keys():
        for k in range(len(net_in.params[l])):
            net_out.params[l][k].data[...] = net_in.params[l][k].data[...]
            net_out.params['l_' + l][k].data[...] = net_in.params[l][k].data[...]
            net_out.params['m_' + l][k].data[...] = net_in.params[l][k].data[...]
            net_out.params['h_' + l][k].data[...] = net_in.params[l][k].data[...]

    # # print for checking
    # layers = ['res4b3_branch2b', 'res4b21_branch2a', 'pooled_rois/centered', 'pooled_rois/pca']
    # for l in layers:
    #     for k in range(len(net_out.params[l])):
    #         print(net_in.params[l][k].data)
    #         print(net_out.params[l][k].data)
    #         print(net_out.params['l_'+l][k].data)
    #     print('\n')

    # save the model
    net_out.save(caffemodel_out)
