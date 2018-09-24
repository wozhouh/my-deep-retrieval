# -*- coding: utf-8 -*-

import os
import argparse

# Python script that add ParamSpec block for training to layers with weights in caffemodel
# but still should revise the data layer and the feature-extraction layer (rmac) by hand


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change the prototxt of ResNet-101 from deploy to train')
    parser.add_argument('--dir', type=str, required=False, help='Path to the workspace')
    parser.add_argument('--old_proto', type=str, required=False, help='Path to the old prototxt file')
    parser.add_argument('--new_proto', type=str, required=False, help='Path to the new prototxt file')
    parser.set_defaults(dir='/home/processyuan/NetworkOptimization/deep-retrieval/')
    parser.set_defaults(old_proto='proto/deploy_resnet101.prototxt')
    parser.set_defaults(new_proto='proto/train_resnet101_pca_template.prototxt')
    args = parser.parse_args()

    f_old = open(os.path.join(args.dir, args.old_proto), 'r')
    f_new = open(os.path.join(args.dir, args.new_proto), 'w')
    lines = f_old.readlines()

    learning_param = '\tparam {\n\t\tlr_mult: 0.0\n\t\tdecay_mult: 0.0\n\t}\n'  # string to be written

    # For ResNet-101, a Convolution layer has 1 learnable param (without bias_term)
    # while a BatchNorm layer has 3 and a Scale layer has 2
    for line in lines:
        f_new.write(line)
        if "Convolution" in line:
            f_new.write(learning_param)
        if "BatchNorm" in line:
            f_new.write(learning_param * 3)
        if "Scale" in line:
            f_new.write(learning_param * 2)

