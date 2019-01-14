# -*- coding: utf-8 -*-

# Python script that tests the model on the Paris dataset by mean precision and mAP

import sys
import caffe
from tqdm import tqdm
sys.path.append(os.path.abspath("../"))
from dataset_helper.paris_helper import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate on the cover dataset')
    parser.add_argument('--gpu', type=int, required=False, help='GPU ID to use (e.g. 0)')
    parser.add_argument('--proto', type=str, required=False, help='Path to the prototxt file')
    parser.add_argument('--weights', type=str, required=False, help='Path to the caffemodel file')
    parser.add_argument('--dataset', type=str, required=False, help='Path to the directory of cover data')
    parser.add_argument('--temp_dir', type=str, required=False,
                        help='Path to a temporary directory to store features and ranking')
    parser.add_argument('--end', type=str, required=False, help='Define the output layer of the net')
    parser.set_defaults(gpu=0)
    parser.set_defaults(proto='/home/processyuan/code/NetworkOptimization/deep-retrieval/'
                              'proto/deploy_resnet101.prototxt')
    parser.set_defaults(weights='/home/processyuan/code/NetworkOptimization/deep-retrieval/'
                                'caffemodel/deep_image_retrieval_model.caffemodel')
    parser.set_defaults(dataset='/home/processyuan/data/Paris')
    parser.set_defaults(temp_dir='/home/processyuan/code/NetworkOptimization/deep-retrieval/eval/temp/')
    parser.set_defaults(end='rmac/normalized')
    args = parser.parse_args()

    # Configure caffe and load the network ResNet-101
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # Load the cover dataset
    pData = ParisDataset(args.dataset)
    pData.get_queries_answer_list()

    # Output of ResNet-101
    output_layer = args.end  # suppose that the layer name is always the same as the blob name
    dim_features = net.blobs[output_layer].data.shape[1]
    features_queries = np.zeros((pData.num_queries, dim_features), dtype=np.float32)
    features_dataset = np.zeros((pData.num_dataset, dim_features), dtype=np.float32)

    # First part, queries
    for k in tqdm(range(pData.num_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
        img, reg = pData.prepare_image_and_grid_regions_for_network('queries', pData.q_fname[k])
        net.blobs['data'].reshape(img.shape[0], int(img.shape[1]), int(img.shape[2]), int(img.shape[3]))
        net.blobs['data'].data[:] = img
        net.blobs['rois'].reshape(reg.shape[0], reg.shape[1])
        net.blobs['rois'].data[:] = reg.astype(np.float32)
        net.forward(end=output_layer)
        features_queries[k] = np.squeeze(net.blobs[output_layer].data)

    # Second part, dataset
    for k in tqdm(range(pData.num_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
        img, reg = pData.prepare_image_and_grid_regions_for_network('dataset', pData.dataset[k])
        net.blobs['data'].reshape(img.shape[0], int(img.shape[1]), int(img.shape[2]), int(img.shape[3]))
        net.blobs['data'].data[:] = img
        net.blobs['rois'].reshape(reg.shape[0], reg.shape[1])
        net.blobs['rois'].data[:] = reg.astype(np.float32)
        net.forward(end=output_layer)
        features_dataset[k] = np.squeeze(net.blobs[output_layer].data)

    # Compute similarity
    sim = features_queries.dot(features_dataset.T)
    # np.save(os.path.join(args.temp_dir, 'sim.npy'), sim)
    # sim = np.load(os.path.join(args.temp_dir, 'sim.npy'))  # test

    # Calculates the precision and mAP
    print('mAP: %f' % pData.cal_mAP(sim))
