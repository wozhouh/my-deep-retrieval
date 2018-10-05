# -*- coding: utf-8 -*-

# Python script that copies the single-pass ResNet-101 to 3-pass and
# adds the param to each layer with 'name', 'lr_mult' and 'decay_mult'

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
    in_layer = False  # the fist line is the name of net

    # used to distinguish different item in branches of teacher network
    branch_name_prefix = ['l_', 'm_', 'h_']
    param_need_prefix = ['bottom', 'top', 'name']
    # param for training
    learning_param = ['\tparam {\n\t\tname: "',
                      '"\n\t\tlr_mult: 0.0\n\t\tdecay_mult: 0.0\n\t}\n']

    for k in range(len(branch_name_prefix)):
        for line in lines:
            # whether in a layer block
            if line.startswith('layer'):
                in_layer = True
            if line.startswith('{'):
                in_layer = False
            layer_block = ""
            new_line = line

            for p in param_need_prefix:
                if p in line:
                    line_temp = line.split('"')
                    new_line = line_temp[0] + '"' + branch_name_prefix[k] + line_temp[1] + '"' + line_temp[2]

            layer_block += new_line

            if 'name' in line:
                layer_name = line.split('"')[1]  # find the layer name

            # For ResNet-101, a Convolution layer has 1 learnable param (without bias_term)
            # while a BatchNorm layer has 3 and a Scale layer has 2
            if 'type' in line:
                if 'Convolution' in line:
                    layer_block += (learning_param[0] + layer_name + '_w' + learning_param[1])
                if 'BatchNorm' in line:
                    layer_block += (learning_param[0] + layer_name + '_1' + learning_param[1])
                    layer_block += (learning_param[0] + layer_name + '_2' + learning_param[1])
                    layer_block += (learning_param[0] + layer_name + '_3' + learning_param[1])
                if 'Scale' in line:
                    layer_block += (learning_param[0] + layer_name + '_1' + learning_param[1])
                    layer_block += (learning_param[0] + layer_name + '_2' + learning_param[1])
                if 'InnerProduct' in line:
                    layer_block += (learning_param[0] + layer_name + '_w' + learning_param[1])
                    layer_block += (learning_param[0] + layer_name + '_b' + learning_param[1])

            f_new.write(layer_block)
