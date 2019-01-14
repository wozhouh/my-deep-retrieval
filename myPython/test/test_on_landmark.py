# -*- coding: utf-8 -*-

# Python script that tests the model on the Landmark dataset by mean precision and mAP

import sys
import os
import caffe
from tqdm import tqdm
import argparse
import cv2
sys.path.append(os.path.abspath("../"))
from region_generator import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate on the cover dataset')
    parser.add_argument('--gpu', type=int, required=False, help='GPU ID to use (e.g. 0)')
    parser.add_argument('--proto', type=str, required=True, help='Path to the prototxt file')
    parser.add_argument('--weights', type=str, required=True, help='Path to the caffemodel file')
    parser.add_argument('--test_dir', type=str, required=False, help='Path to the directory of queries and dataset')
    parser.add_argument('--end', type=str, required=False, help='Define the output layer of the net')
    parser.add_argument('--temp_dir', type=str, required=False, help='Path to save the sim.npy')
    parser.set_defaults(gpu=0)
    parser.set_defaults(test_dir='/home/processyuan/data/Landmark/cls/training-test/')
    parser.set_defaults(end='rmac/normalized')
    parser.set_defaults(temp_dir='/home/processyuan/code/NetworkOptimization/deep-retrieval/eval/temp/')
    args = parser.parse_args()

    means = np.array([103.93900299,  116.77899933,  123.68000031], dtype=np.float32)[:, None, None]

    # Prepare the queries and the dataset
    queries_dir = os.path.join(args.test_dir, "queries")
    dataset_dir = os.path.join(args.test_dir, "dataset")
    cls_list = os.listdir(queries_dir)
    q_path = []  # path to the images as queries
    a_path = []  # path to the images in the dataset
    q_cls = []  # class of each query
    a_cls = []  # class of each image in the dataset
    a_num = []  # number of images in the dataset
    for c in cls_list:
        q_cls_path = os.path.join(queries_dir, c)
        a_cls_path = os.path.join(dataset_dir, c)
        for j in os.listdir(q_cls_path):
            q_img_path = os.path.join(q_cls_path, j)
            q_path.append(q_img_path)
            q_cls.append(c)
            a_num.append(len(os.listdir(os.path.join(dataset_dir, c))))
        for i in os.listdir(a_cls_path):
            a_img_path = os.path.join(a_cls_path, i)
            a_path.append(a_img_path)
            a_cls.append(c)
    num_queries = len(q_cls)
    num_dataset = len(a_cls)

    # Configure caffe and load the network ResNet-101
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # Output of ResNet-101
    output_layer = args.end  # suppose that the layer name is always the same as the blob name
    dim_features = net.blobs[output_layer].data.shape[1]
    features_queries = np.zeros((num_queries, dim_features), dtype=np.float32)
    features_dataset = np.zeros((num_dataset, dim_features), dtype=np.float32)

    # First part, queries
    for k in tqdm(range(num_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
        img_path = q_path[k]
        img_temp = cv2.imread(img_path).transpose(2, 0, 1) - means
        all_regions = [get_rmac_region_coordinates(img_temp.shape[1], img_temp.shape[2], 2)]
        reg = pack_regions_for_network(all_regions)
        img = np.expand_dims(img_temp, axis=0)
        net.blobs['data'].reshape(img.shape[0], int(img.shape[1]), int(img.shape[2]), int(img.shape[3]))
        net.blobs['data'].data[:] = img
        net.blobs['rois'].reshape(reg.shape[0], reg.shape[1])
        net.blobs['rois'].data[:] = reg.astype(np.float32)
        net.forward(end=output_layer)
        features_queries[k] = np.squeeze(net.blobs[output_layer].data)

    # Second part, dataset
    for k in tqdm(range(num_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
        img_path = a_path[k]
        img_temp = cv2.imread(img_path).transpose(2, 0, 1) - means
        all_regions = [get_rmac_region_coordinates(img_temp.shape[1], img_temp.shape[2], 2)]
        reg = pack_regions_for_network(all_regions)
        img = np.expand_dims(img_temp, axis=0)
        net.blobs['data'].reshape(img.shape[0], int(img.shape[1]), int(img.shape[2]), int(img.shape[3]))
        net.blobs['data'].data[:] = img
        net.blobs['rois'].reshape(reg.shape[0], reg.shape[1])
        net.blobs['rois'].data[:] = reg.astype(np.float32)
        net.forward(end=output_layer)
        features_dataset[k] = np.squeeze(net.blobs[output_layer].data)

    # Compute similarity
    sim = features_queries.dot(features_dataset.T)
    # np.save(os.path.join(args.temp_dir, "queries.npy"), features_queries)
    # np.save(os.path.join(args.temp_dir, "dataset.npy"), features_dataset)
    # np.save(os.path.join(args.temp_dir, "sim.npy"), sim)
    # sim = np.load(args.sim_npy)  # Debug

    # Calculates the value of mAP according to the standard of VOC2010 and later
    q_AP = np.zeros(num_queries, dtype=np.float32)
    idx = np.argsort(sim, axis=1)[:, ::-1]
    for q in range(num_queries):
        cnt_correct_last = 0
        recall = np.zeros(a_num[q], dtype=np.float32)
        for d in range(num_dataset):
            top_k = d + 1
            top_idx = list(idx[q, :top_k])
            cnt_correct = len([i for i in top_idx if a_cls[i] == q_cls[q]])
            assert cnt_correct >= cnt_correct_last
            assert cnt_correct <= a_num[q]
            if cnt_correct > cnt_correct_last:
                recall[cnt_correct - 1] = float(cnt_correct) / float(top_k)  # precision under the given recall
                cnt_correct_last = cnt_correct
            if cnt_correct == a_num[q]:
                break

        # calculates the maximum precision when no less than given recall
        recall_max = np.zeros(a_num[q], dtype=np.float32)
        for r in range(a_num[q]):
            recall_max[r] = np.max(recall[r:])
        q_AP[q] = recall_max.mean(axis=0)
        print("AP of query %d: %f" % (q, q_AP[q]))

    mAP = q_AP.mean(axis=0) * 100.0
    print("mAP: %f" % mAP)
