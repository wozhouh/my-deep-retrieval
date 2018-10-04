# -*- coding: utf-8 -*-

# Python script that removes the learning params from each layer in the prototxt

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
    in_param = False

    for line in lines:
        if (not in_param) and line.startswith('\tparam {'):
            in_param = True
        if not in_param:
            f_new.write(line)
        if in_param and line.startswith('\t}'):
            in_param = False

    f_old.close()
    f_new.close()
