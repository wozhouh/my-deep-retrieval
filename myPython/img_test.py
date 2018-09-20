# -*- coding: utf-8 -*-

import argparse
import cv2
from class_helper import *

if __name__ == '__main__':

    # Config
    parser = argparse.ArgumentParser(description='Evaluate Oxford')
    parser.add_argument('--S', type=int, required=False, help='Resize larger side of image to S pixels (e.g. 800)')
    parser.add_argument('--L', type=int, required=False, help='Use L spatial levels (e.g. 2)')
    parser.add_argument('--dataset', type=str, required=False, help='Path to the Oxford / Paris directory')
    parser.set_defaults(S=512)
    parser.set_defaults(L=2)
    parser.set_defaults(dataset='/home/processyuan/data/Oxford/')
    args = parser.parse_args()

    S = args.S
    L = args.L

    # Load the dataset and the image helper
    dataset = Dataset(args.dataset)
    image_helper = ImageHelper(S, L)

    N_queries = dataset.N_queries
    N_dataset = dataset.N_images
    num_test = 30
    min_num = 100

    for i in range(N_dataset):
        # I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_query_filename(i),
        #                                                                roi=dataset.get_query_roi(i))
        im = cv2.imread(dataset.get_filename(i))
        im_size_hw = np.array(im.shape[0:2])
        ratio = float(S) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        im_resized = cv2.resize(im, (new_size[1], new_size[0]))

        # roi = np.round(dataset.get_query_roi(i) * ratio).astype(np.int32)
        # im_resized = im_resized[roi[1]:roi[3], roi[0]:roi[2], :]  # crop

        # cv2.imwrite('/home/processyuan/NetworkOptimization/deep-retrieval/myTest/' + str(i) + '.jpg', im)


        all_regions = [image_helper.get_rmac_region_coordinates(im_resized.shape[0], im_resized.shape[1], L)]
        R = image_helper.pack_regions_for_network(all_regions)
        if min_num > R.shape[0]:
            min_num = R.shape[0]
        print("shape: %d  min: %d" % (R.shape[0], min_num))

        # for k in range(R.shape[0]):
        #     cv2.rectangle(im_resized, (R[k, 1], R[k, 2]), (R[k, 3], R[k, 4]), (0, 255, 0), 3)
        #
        # cv2.imwrite('/home/processyuan/NetworkOptimization/deep-retrieval/myTest/' + str(i) + '_' + str(S)
        #             + '_im_resized.jpg', im_resized)
