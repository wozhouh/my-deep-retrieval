# -*- coding: utf-8 -*-

# Python script that tests the model on the cover dataset by mean precision and mAP

import sys
import caffe
from tqdm import tqdm
from cover_helper import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate on the cover dataset')
    parser.add_argument('--gpu', type=int, required=False, help='GPU ID to use (e.g. 0)')
    parser.add_argument('--proto', type=str, required=False, help='Path to the prototxt file')
    parser.add_argument('--weights', type=str, required=False, help='Path to the caffemodel file')
    parser.add_argument('--dataset', type=str, required=False, help='Path to the directory of cover data')
    parser.add_argument('--temp_dir', type=str, required=False,
                        help='Path to a temporary directory to store features and ranking')
    parser.add_argument('--end', type=str, required=False, help='Define the output layer of the net')
    parser.add_argument('--multires', dest='multires', action='store_true', help='Enable multiresolution features')
    parser.set_defaults(gpu=0)
    parser.set_defaults(proto='/home/processyuan/code/NetworkOptimization/deep-retrieval/'
                              'proto/deploy_resnet101.prototxt')
    parser.set_defaults(weights='/home/processyuan/code/NetworkOptimization/deep-retrieval/'
                                'caffemodel/deep_image_retrieval_model.caffemodel')
    parser.set_defaults(dataset='/home/processyuan/data/cover')
    parser.set_defaults(temp_dir='/home/processyuan/code/NetworkOptimization/deep-retrieval/eval/temp/')
    parser.set_defaults(end='rmac/normalized')
    parser.set_defaults(multires=False)
    args = parser.parse_args()

    # Configure caffe and load the network ResNet-101
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # Load the cover dataset
    cData = CoverDataset(args.dataset)
    cData.get_queries_answer_list()

    # Output of ResNet-101
    output_layer = args.end  # suppose that the layer name is always the same as the blob name
    dim_features = net.blobs[output_layer].data.shape[1]
    features_queries = np.zeros((cData.num_queries, dim_features), dtype=np.float32)
    features_dataset = np.zeros((cData.num_dataset, dim_features), dtype=np.float32)

    Ss = [496] if not args.multires else [248, 496, 744]  # multi-resolution of (256, 512, 768)

    # First part, queries
    for S in Ss:
        for k in tqdm(range(cData.num_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
            cData.S = S
            img, reg = cData.prepare_image_and_grid_regions_for_network('queries', cData.q_fname[k])
            net.blobs['data'].reshape(img.shape[0], int(img.shape[1]), int(img.shape[2]), int(img.shape[3]))
            net.blobs['data'].data[:] = img
            net.blobs['rois'].reshape(reg.shape[0], reg.shape[1])
            net.blobs['rois'].data[:] = reg.astype(np.float32)
            net.forward(end=output_layer)
            features_queries[k] = np.squeeze(net.blobs[output_layer].data)
        features_queries_fname = os.path.join(args.temp_dir, "queries_S{0}.npy".format(S))
        np.save(features_queries_fname, features_queries)
    features_queries = np.dstack(
        [np.load(os.path.join(args.temp_dir, "queries_S{0}.npy".format(S))) for S in Ss]).sum(axis=2)
    # np.save(os.path.join(args.temp_dir, 'queries_baseline.npy'), features_queries)

    # Second part, dataset
    for S in Ss:
        for k in tqdm(range(cData.num_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
            cData.S = S
            img, reg = cData.prepare_image_and_grid_regions_for_network('dataset', cData.dataset[k])
            net.blobs['data'].reshape(img.shape[0], int(img.shape[1]), int(img.shape[2]), int(img.shape[3]))
            net.blobs['data'].data[:] = img
            net.blobs['rois'].reshape(reg.shape[0], reg.shape[1])
            net.blobs['rois'].data[:] = reg.astype(np.float32)
            net.forward(end=output_layer)
            features_dataset[k] = np.squeeze(net.blobs[output_layer].data)
        features_dataset_fname = os.path.join(args.temp_dir, "dataset_S{0}.npy".format(S))
        np.save(features_dataset_fname, features_dataset)
    features_dataset = np.dstack(
        [np.load(os.path.join(args.temp_dir, "dataset_S{0}.npy".format(S))) for S in Ss]).sum(axis=2)
    # np.save(os.path.join(args.temp_dir, 'dataset_baseline.npy'), features_dataset)

    # Compute similarity
    sim = features_queries.dot(features_dataset.T)
    np.save(os.path.join(args.temp_dir, 'sim.npy'), sim)
    # sim = np.load(os.path.join(args.temp_dir, 'sim.npy'))  # test

    # Calculates the precision and mAP
    print('precision: %f' % cData.cal_precision(sim, output_img=True))
    print('mAP: %f' % cData.cal_mAP(sim))
