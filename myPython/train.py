# -*- coding: utf-8 -*-

# Python script that starts the Caffe training

import argparse
import caffe


if __name__ == "__main__":
    # configure
    parser = argparse.ArgumentParser(description='load the model to start training on Caffe')
    parser.add_argument('--proto', type=str, required=True, help='Path to the prototxt file')
    parser.add_argument('--weights', type=str, required=True, help='Path to the caffemodel file')
    parser.add_argument('--gpu', type=int, required=False, default=0, help='index of Used GPU')
    args = parser.parse_args()

    # setting
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

    # create the solver
    solver = caffe.net.copy_from(args.proto)

    # load the caffemodel
    solver.net.copy_from(args.weights)

    # start training
    solver.solve()
