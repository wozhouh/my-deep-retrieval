# -*- coding: utf-8 -*-

# Python script that adds the learning params to each layer in the prototxt
# That is, the block surrounded by 'param{}'

import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change the prototxt of ResNet-101 from deploy to train')
    parser.add_argument('--old_proto', type=str, required=False, help='Path to the old prototxt file')
    parser.add_argument('--new_proto', type=str, required=False, help='Path to the new prototxt file')
    parser.set_defaults(old_proto='/home/processyuan/NetworkOptimization/deep-retrieval/proto/deploy_resnet101.prototxt')
    parser.set_defaults(new_proto='/home/processyuan/NetworkOptimization/deep-retrieval/'
                                  'proto/train_resnet101_distilling_template.prototxt')
    args = parser.parse_args()

    f_old = open(os.path.join(args.old_proto), 'r')
    f_new = open(os.path.join(args.new_proto), 'w')
    lines = f_old.readlines()

    learning_param = '\tparam {\n\t\tlr_mult: 0.0\n\t\tdecay_mult: 0.0\n\t}\n'

    for line in lines:
        f_old = f_new.write(line)
        # For ResNet-101, a Convolution layer has 1 learnable param (without bias_term)
        # while a BatchNorm layer has 3 and a Scale layer has 2
        if 'type' in line:
            if 'Convolution' in line:
                f_new.write(learning_param)
            if 'BatchNorm' in line:
                f_new.write(learning_param * 3)
            if 'Scale' in line:
                f_new.write(learning_param * 2)
            if 'InnerProduct' in line:
                f_new.write(learning_param * 2)
