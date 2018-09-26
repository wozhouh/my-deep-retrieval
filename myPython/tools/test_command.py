# -*- coding: utf-8 -*-

# Python script for Linux command line debugging
# Config the file path as you wish

import numpy as np
import caffe
# from class_helper import *

PROTO = '/home/processyuan/NetworkOptimization/deep-retrieval/proto/train-distilling/deploy_resnet101_student.prototxt'
MODEL = '/home/processyuan/NetworkOptimization/deep-retrieval/caffemodel/deep_image_retrieval_model.caffemodel'
DATASET = '/home/processyuan/data/Oxford/'

S = 512
L = 2
caffe.set_device(0)
caffe.set_mode_cpu()

net = caffe.Net(PROTO, MODEL, caffe.TEST)

dataset = Dataset(DATASET)
image_helper = ImageHelper(S, L)


I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_query_filename(3),
                                                               roi=dataset.get_query_roi(3))

net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
net.blobs['data'].data[:] = I
net.blobs['rois'].reshape(R.shape[0], R.shape[1])
net.blobs['rois'].data[:] = R.astype(np.float32)
net.forward()
