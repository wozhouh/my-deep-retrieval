# -*- coding: utf-8 -*-

# Python class for modifying the model (prototxt and caffemodel)

import argparse
import caffe


class ModelTools:
    def __init__(self, proto, weights, gpu):
        self.proto = proto
        self.weights = weights
        # open the prototxt
        f_proto = open(self.proto, 'r')
        self.lines = f_proto.readlines()
        # build the net
        self.net = caffe.Net(self.proto, self.weights, caffe.TEST)
        # setting
        caffe.set_mode_gpu()
        caffe.set_device(gpu)
        # other params
        self.learning_params = '\tparam {\n\t\tlr_mult: 0.0\n\t\tdecay_mult: 0.0\n\t}\n'  # to stop back-propagation

    # print the shape of all the weight blobs stored in .caffemodel file
    def check_blob_shape(self):
        for layer in self.net.params.keys():
            print(layer)
            for dim in range(len(self.net.params[layer])):
                print(self.net.params[layer][dim].data.shape)

    # print the datas of all the weight blobs in the list of layers
    # param 'layers' is a list of layer names
    def check_blob_data(self, layers):
        for l in layers:
            for k in range(len(self.net.params[l])):
                print(self.net.params[l][k].data)
            print('\n')

    # compare which layers in the two models are different
    def compare_model(self, other_proto):
        other_net = caffe.Net(other_proto, args.weights, caffe.TEST)
        # compare the weights
        print("weights that missing: ")
        for l in other_net.params.keys():
            if l not in self.net.params.keys():
                print(l)
        print("weights that additional: ")
        for l in self.net.params.keys():
            if l not in other_net.params.keys():
                print(l)

    # Remove the learning params to each layer in the prototxt, that is, the block surrounded by 'param{}'
    def remove_learning_params(self, new_proto):
        f_new_proto = open(new_proto, 'w')
        in_param = False
        for line in self.lines:
            if (not in_param) and line.startswith('\tparam {'):
                in_param = True
            if not in_param:
                f_new_proto.write(line)
            if in_param and line.startswith('\t}'):
                in_param = False
        f_new_proto.close()

    # Add the learning params to each layer for later training
    # layers within lines lower than 'th' will add learning-params that lr_mult=0 to stop back propagation
    # layers within lines higher than 'th' will not change
    def add_learning_params(self, new_proto, th):
        f_new_proto = open(new_proto, 'w')
        # a Convolution layer should add 1 learning param (without bias_term)
        # For a BatchNorm layer, the 'use_global_stats' should be changed to 'false'
        # the Scale layer and the ReLU layer will not change
        for line in self.lines:
            if 'use_global_stats' in line:
                # replace 'true' with 'false'
                split_temp = line.split('true')
                f_new_proto.write(split_temp[0] + 'false' + split_temp[-1])
            else:
                f_new_proto.write(line)
            if 'Convolution' in line:
                f_new_proto.write(self.learning_params)

        f_new_proto.close()

    # Copy the single-pass ResNet-101 to 3-pass and
    # add the learning params to each layer with 'name', 'lr_mult' and 'decay_mult'
    def make_teacher_network(self, new_proto):
        f_new_proto = open(new_proto, 'w')
        # used to distinguish different branches of teacher network
        branch_name_prefix = ['l_', 'm_', 'h_']
        param_need_prefix = ['bottom', 'top', 'name']
        # param for training
        learning_param = ['\tparam {\n\t\tname: "',
                          '"\n\t\tlr_mult: 0.0\n\t\tdecay_mult: 0.0\n\t}\n']

        for k in range(len(branch_name_prefix)):
            layer_name = ''
            for line in self.lines:
                new_line = line
                for p in param_need_prefix:
                    if p in line:
                        line_temp = line.split('"')
                        new_line = line_temp[0] + '"' + branch_name_prefix[k] + line_temp[1] + '"' + line_temp[2]

                f_new_proto.write(new_line)

                if 'name' in line:
                    layer_name = line.split('"')[1]  # find the layer name

                # For ResNet-101, a Convolution layer has 1 learnable param (without bias_term)
                # while a BatchNorm layer has 3 and a Scale layer has 2
                if 'type' in line:
                    if 'Convolution' in line:
                        f_new_proto.write(learning_param[0] + layer_name + '_w' + learning_param[1])
                    if 'BatchNorm' in line:
                        f_new_proto.write(learning_param[0] + layer_name + '_1' + learning_param[1])
                        f_new_proto.write(learning_param[0] + layer_name + '_2' + learning_param[1])
                        f_new_proto.write(learning_param[0] + layer_name + '_3' + learning_param[1])
                    if 'Scale' in line:
                        f_new_proto.write(learning_param[0] + layer_name + '_1' + learning_param[1])
                        f_new_proto.write(learning_param[0] + layer_name + '_2' + learning_param[1])
                    if 'InnerProduct' in line:
                        f_new_proto.write(learning_param[0] + layer_name + '_w' + learning_param[1])
                        f_new_proto.write(learning_param[0] + layer_name + '_b' + learning_param[1])

        f_new_proto.close()

    # copy the weights of 1-pass network to 3-pass teacher network
    # Run after running the 'make_teacher_network()'
    def save_teacher_network_weights(self, teacher_proto, caffemodel_path):
        teacher_net = caffe.Net(teacher_proto, self.weights, caffe.TEST)
        for l in self.net.params.keys():
            for k in range(len(self.net.params[l])):
                teacher_net.params[l][k].data[...] = self.net.params[l][k].data[...]
                teacher_net.params['l_' + l][k].data[...] = self.net.params[l][k].data[...]
                teacher_net.params['m_' + l][k].data[...] = self.net.params[l][k].data[...]
                teacher_net.params['h_' + l][k].data[...] = self.net.params[l][k].data[...]
        # save the model
        teacher_net.save(caffemodel_path)


if __name__ == "__main__":
    # configure
    parser = argparse.ArgumentParser(description='print the shape of weights stored in caffemodel')
    parser.add_argument('--proto', type=str, required=True, help='Path to the prototxt file')
    parser.add_argument('--weights', type=str, required=True, help='Path to the caffemodel file')
    parser.add_argument('--gpu', type=str, required=False, help='index of Used GPU')
    parser.set_defaults(gpu=0)
    args = parser.parse_args()

    # init
    model_tools = ModelTools(args.proto, args.weights, args.gpu)

    # comparison
    model_tools.compare_model(other_proto='/home/gordonwzhe/code/my-deep-retrieval/proto/'
                                          'deploy_resnet101.prototxt')
    # model_tools.compare_model(other_proto='/home/gordonwzhe/code/my-deep-retrieval/proto/'
    #                           'distilling/deploy_resnet101_student.prototxt')

    # deploy to train
    model_tools.add_learning_params(new_proto='/home/gordonwzhe/code/my-deep-retrieval/proto/'
                                              'distilling/train_resnet101_paris.prototxt', th=10000)