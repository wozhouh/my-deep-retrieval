# -*- coding: utf-8 -*-

import os
import argparse


def get_param_to_prototxt(type):
    param = '\tparam {\n\t\tlr_mult: 0.0\n\t\tdecay_mult: 0.0\n\t}\n'  # string to be written
    if type == 'conv':
        return param
    if type == 'bn':
        return param * 3
    if type == 'scale':
        return param * 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change the prototxt of ResNet-101 from deploy to train')
    parser.add_argument('--dir', type=str, required=False, help='Path to the workspace')
    parser.add_argument('--old_proto', type=str, required=False, help='Path to the old prototxt file')
    parser.add_argument('--new_proto', type=str, required=False, help='Path to the new prototxt file')
    parser.set_defaults(dir='/home/processyuan/NetworkOptimization/deep-retrieval/')
    parser.set_defaults(old_proto='proto/train_resnet101_template.prototxt')
    parser.set_defaults(new_proto='proto/train_resnet101_normpython.prototxt')
    args = parser.parse_args()

    f_old = open(os.path.join(args.dir, args.old_proto), 'r')
    f_new = open(os.path.join(args.dir, args.new_proto), 'w')

    lines = f_old.readlines()
    for line in lines:
        f_new.write(line)
        if line.startswith('\ttype: "Convolution"'):
            f_new.write(get_param_to_prototxt('conv'))
        if line.startswith('\ttype: "BatchNorm"'):
            f_new.write(get_param_to_prototxt('bn'))
        if line.startswith('\ttype: "Scale"'):
            f_new.write(get_param_to_prototxt('scale'))
