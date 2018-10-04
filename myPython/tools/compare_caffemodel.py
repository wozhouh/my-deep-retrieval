# -*- coding: utf-8 -*-

# Python script to compare whether two net described in prototxt are different from each other

import caffe
import argparse

if __name__ == "__main__":

    # configure
    parser = argparse.ArgumentParser(description='print the shape of weights stored in caffemodel')
    parser.add_argument('--old_proto', type=str, required=True, help='Path to the old prototxt file')
    parser.add_argument('--new_proto', type=str, required=True, help='Path to the new prototxt file')
    parser.add_argument('--weights', type=str, required=True, help='Path to the caffemodel file')
    args = parser.parse_args()

    # setting
    caffe.set_mode_cpu()

    # build the net
    net_new = caffe.Net(args.new_proto, args.weights, caffe.TEST)
    net_old = caffe.Net(args.old_proto, args.weights, caffe.TEST)

    # compare the weights
    for l in net_new.params.keys():
        if l not in net_old.params.keys():
            print(l)
