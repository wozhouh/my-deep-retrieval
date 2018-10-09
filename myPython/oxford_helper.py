# -*- coding: utf-8 -*-

# Python class with common operations on the Oxford dataset

import numpy as np
import os
import argparse
import cv2
import random
import subprocess
from collections import OrderedDict

# resize the image so that the longer side equals the given size
def transform_image(size, src, dst):
    im = cv2.imread(src)
    im_size_hw = np.array(im.shape[0:2])
    ratio = float(size) / np.max(im_size_hw)
    new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
    im_resized = cv2.resize(im, (new_size[1], new_size[0]))
    cv2.imwrite(dst, im_resized)


class ImageHelper:
    def __init__(self, S, L):
        self.S = S
        self.L = L
        # Load and reshape the means to subtract to the inputs
        self.means = np.array([103.93900299,  116.77899933,  123.68000031], dtype=np.float32)[None, :, None, None]

    def prepare_image_and_grid_regions_for_network(self, fname, roi=None):
        # Extract image, resize at desired size, and extract roi region if
        # available. Then compute the rmac grid in the net format: ID X Y W H
        I, im_resized = self.load_and_prepare_image(fname, roi)
        if self.L == 0:
            # Encode query in mac format instead of rmac, so only one region
            # Regions are in ID X Y W H format
            R = np.zeros((1, 5), dtype=np.float32)
            R[0, 3] = im_resized.shape[1] - 1
            R[0, 4] = im_resized.shape[0] - 1
        else:
            # Get the region coordinates and feed them to the network.
            all_regions = [self.get_rmac_region_coordinates(im_resized.shape[0], im_resized.shape[1], self.L)]
            R = self.pack_regions_for_network(all_regions)
        return I, R

    def get_rmac_features(self, I, R, net, end_layer=None):
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        if end_layer is None:
            net.forward()
        else:
            net.forward(end=end_layer)
        return np.squeeze(net.blobs[end_layer].data)

    def load_and_prepare_image(self, fname, roi=None):
        # Read image, get aspect ratio, and resize such as the largest side equals S
        im = cv2.imread(fname)
        im_size_hw = np.array(im.shape[0:2])
        ratio = float(self.S) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        im_resized = cv2.resize(im, (new_size[1], new_size[0]))
        # If there is a roi, adapt the roi to the new size and crop. Do not rescale
        # the image once again
        if roi is not None:
            roi = np.round(roi * ratio).astype(np.int32)
            im_resized = im_resized[roi[1]:roi[3], roi[0]:roi[2], :]  # crop
        # Transpose for network and subtract mean
        I = im_resized.transpose(2, 0, 1) - self.means  # H x W x 3 -> 3 x H x W
        return I, im_resized

    def pack_regions_for_network(self, all_regions):
        n_regs = np.sum([len(e) for e in all_regions])
        R = np.zeros((n_regs, 5), dtype=np.float32)
        cnt = 0
        # There should be a check of overflow...
        for i, r in enumerate(all_regions):
            try:
                R[cnt:cnt + r.shape[0], 0] = i
                R[cnt:cnt + r.shape[0], 1:] = r
                cnt += r.shape[0]
            except:
                continue
        assert cnt == n_regs
        R = R[:n_regs]
        # regs where in xywh format. R is in xyxy format, where the last coordinate is included. Therefore...
        R[:n_regs, 3] = R[:n_regs, 1] + R[:n_regs, 3] - 1
        R[:n_regs, 4] = R[:n_regs, 2] + R[:n_regs, 4] - 1
        return R

    def get_rmac_region_coordinates(self, H, W, L):
        # Almost verbatim from Tolias et al Matlab implementation.
        # Could be heavily pythonized, but really not worth it...
        # Desired overlap of neighboring regions
        ovr = 0.4
        # Possible regions for the long dimension
        steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
        w = np.minimum(H, W)

        b = (np.maximum(H, W) - w) / (steps - 1)
        # steps(idx) regions for long dimension. The +1 comes from Matlab
        # 1-indexing...
        idx = np.argmin(np.abs(((w ** 2 - w * b) / w ** 2) - ovr)) + 1

        # Region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx
        elif H > W:
            Hd = idx

        regions_xywh = []
        for l in range(1, L + 1):
            wl = np.floor(2 * w / (l + 1))
            wl2 = np.floor(wl / 2 - 1)
            # Center coordinates
            if l + Wd - 1 > 0:
                b = (W - wl) / (l + Wd - 1)
            else:
                b = 0
            cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
            # Center coordinates
            if l + Hd - 1 > 0:
                b = (H - wl) / (l + Hd - 1)
            else:
                b = 0
            cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

            for i_ in cenH:
                for j_ in cenW:
                    regions_xywh.append([j_, i_, wl, wl])

        # Round the regions. Careful with the borders!
        for i in range(len(regions_xywh)):
            for j in range(4):
                regions_xywh[i][j] = int(round(regions_xywh[i][j]))
            if regions_xywh[i][0] + regions_xywh[i][2] > W:
                regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W)
            if regions_xywh[i][1] + regions_xywh[i][3] > H:
                regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H)
        return np.array(regions_xywh).astype(np.float32)


class OxfordDataset:
    def __init__(self, path):
        self.path = path
        # Parse the label files. Some challenges as filenames do not correspond
        # exactly to query names. Go through all the labels to:
        # i) map names to filenames and vice versa
        # ii) get the relevant regions of interest of the queries,
        # iii) get the indexes of the dataset images that are queries
        # iv) get the relevants / non-relevants list

        self.name_to_filename = OrderedDict()
        self.filename_to_name = {}
        self.non_relevants = {}
        self.junk = {}
        self.relevants = {}
        self.q_roi = {}
        # Load the dataset GT
        self.lab_root = '{0}/lab/'.format(self.path)
        self.img_root = '{0}/jpg/'.format(self.path)
        # Get the filenames without the extension
        self.img_filenames = [e[:-4] for e in np.sort(os.listdir(self.img_root))]
        self.load()

    def load(self):
        lab_filenames = np.sort(os.listdir(self.lab_root))
        for e in lab_filenames:
            if e.endswith('_query.txt'):
                q_name = e[:-len('_query.txt')]
                q_data = file("{0}/{1}".format(self.lab_root, e)).readline().split(" ")
                q_filename = q_data[0][5:] if q_data[0].startswith('oxc1_') else q_data[0]
                self.filename_to_name[q_filename] = q_name
                self.name_to_filename[q_name] = q_filename
                good = set([e.strip() for e in file("{0}/{1}_ok.txt".format(self.lab_root, q_name))])
                good = good.union(set([e.strip() for e in file("{0}/{1}_good.txt".format(self.lab_root, q_name))]))
                junk = set([e.strip() for e in file("{0}/{1}_junk.txt".format(self.lab_root, q_name))])
                good_plus_junk = good.union(junk)
                self.relevants[q_name] = [i for i in range(len(self.img_filenames)) if self.img_filenames[i] in good]
                self.junk[q_name] = [i for i in range(len(self.img_filenames)) if self.img_filenames[i] in junk]
                self.non_relevants[q_name] = [i for i in range(len(self.img_filenames)) if
                                              self.img_filenames[i] not in good_plus_junk]
                self.q_roi[q_name] = np.array(map(float, q_data[1:]), dtype=np.float32)

        self.q_names = self.name_to_filename.keys()
        self.q_index = np.array([self.img_filenames.index(self.name_to_filename[qn]) for qn in self.q_names])
        self.N_images = len(self.img_filenames)
        self.N_queries = len(self.q_index)

    def score(self, sim, temp_dir, eval_bin):
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        idx = np.argsort(sim, axis=1)[:, ::-1]
        maps = [self.score_rnk_partial(i, idx[i], temp_dir, eval_bin) for i in range(len(self.q_names))]
        for i in range(len(self.q_names)):
            print "{0}: {1:.2f}".format(self.q_names[i], 100 * maps[i])
        print 20 * "-"
        print "Mean: {0:.2f}".format(100 * np.mean(maps))

    def score_rnk_partial(self, i, idx, temp_dir, eval_bin):
        rnk = np.array(self.img_filenames)[idx]
        with open("{0}/{1}.rnk".format(temp_dir, self.q_names[i]), 'w') as f:
            f.write("\n".join(rnk) + "\n")
        cmd = "{0} {1}{2} {3}/{4}.rnk".format(eval_bin, self.lab_root, self.q_names[i], temp_dir, self.q_names[i])
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        map_ = float(p.stdout.readlines()[0])
        p.wait()
        return map_

    def get_filename(self, i):
        return os.path.normpath("{0}/{1}.jpg".format(self.img_root, self.img_filenames[i]))

    def get_query_filename(self, i):
        return os.path.normpath("{0}/{1}.jpg".format(self.img_root, self.img_filenames[self.q_index[i]]))

    def get_query_roi(self, i):
        return self.q_roi[self.q_names[i]]

    def make_training_test_set(self, training_dir, test_dir, img_size, training_ratio=0.5):
        img_dir = 'jpg'
        lab_dir = 'lab'
        img_all = []  # list of all image files: 5063 images in total
        img_root = os.path.join(self.path, img_dir)  # Path to images of original dataset
        lab_root = os.path.join(self.path, lab_dir)  # Path to txt of original dataset
        q_filename_temp = self.filename_to_name.keys()
        q_filename = [q_filename_temp[k] + '.jpg' for k in range(self.N_queries)]
        # delete first
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
            os.makedirs(os.path.join(training_dir, img_dir))
            os.makedirs(os.path.join(training_dir, lab_dir))
        else:
            train_jpg_path = os.path.join(training_dir, img_dir)
            for f in os.listdir(train_jpg_path):
                full_name = os.path.join(train_jpg_path, f)
                if os.path.isfile(full_name):
                    os.remove(full_name)

        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            os.makedirs(os.path.join(test_dir, img_dir))
            os.makedirs(os.path.join(test_dir, lab_dir))
        else:
            test_jpg_path = os.path.join(test_dir, img_dir)
            for f in os.listdir(test_jpg_path):
                full_name = os.path.join(test_jpg_path, f)
                if os.path.isfile(full_name):
                    os.remove(full_name)

        # get the list of all image files
        for f in os.listdir(img_root):
            img_all.append(f)

        random.shuffle(img_all)  # shuffle
        img_num = len(img_all)
        training_num = int(img_num * training_ratio)
        img_training = img_all[0: training_num]
        img_test = img_all[training_num:]

        # make the "jpg" directory of training set (Note that images in queries is shared by both training and test set)
        for k in range(len(img_training)):
            src_file = os.path.join(img_root, img_training[k])
            dst_file = os.path.join(training_dir, img_dir, img_training[k])
            if os.path.isfile(src_file):
                if img_training[k] in q_filename:
                    img_test.append(img_training[k])
                else:
                    transform_image(img_size, src_file, dst_file)
            else:
                print('image not found: %s' % src_file)

        # make the "jpg" directory of test set
        for k in range(len(img_test)):
            src_file = os.path.join(img_root, img_test[k])
            dst_file = os.path.join(test_dir, img_dir, img_test[k])
            if os.path.isfile(src_file):
                if img_test[k] in q_filename:
                    q_file = os.path.join(training_dir, img_dir, img_test[k])
                    open(q_file, 'wb').write(open(src_file, 'rb').read())  # save the origin image
                    open(dst_file, 'wb').write(open(src_file, 'rb').read())
                    if img_test[k] not in img_training:
                        img_training.append(img_test[k])
                else:
                    transform_image(img_size, src_file, dst_file)
            else:
                print('image not found: %s' % src_file)

        # make the "lab" directory
        for f in os.listdir(lab_root):
            full_name = os.path.join(lab_root, f)
            f_lab = open(full_name, 'r')
            f_train_lab = open(os.path.join(training_dir, lab_dir, f), 'w')
            f_test_lab = open(os.path.join(test_dir, lab_dir, f), 'w')
            if f.endswith('_query.txt'):
                content = f_lab.read()
                f_train_lab.write(content)
                f_test_lab.write(content)
            else:
                for line in f_lab.readlines():
                    img_filename = line.strip() + '.jpg'
                    if img_filename in img_training:
                        f_train_lab.write(line)
                    if img_filename in img_test:
                        f_test_lab.write(line)

    # Calculates each channel's mean value of RGB images in the training set
    # for Oxford dataset, the mean value for each channel is [99.74151, 108.75074, 113.17747]
    def cal_image_mean_channel(self):
        img_sum = np.zeros(3, dtype=np.float32)
        fname = os.listdir(self.img_root)
        for f in fname:
            img = cv2.imread(os.path.join(self.img_root, f))
            img_sum += img.mean(axis=0).mean(axis=0)
        return img_sum / len(fname)

    # make a dataset with the same images in which the resolution is united
    def uni_dataset(self, new_path, img_h, img_w):
        new_jpg_dir = os.path.join(new_path, 'jpg')
        new_lab_dir = os.path.join(new_path, 'lab')
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            os.makedirs(new_jpg_dir)
            os.makedirs(new_lab_dir)
        # transform the images
        for i in self.img_root:
            img_src_path = os.path.join(self.img_root, i)
            img_dst_path = os.path.join(new_jpg_dir, i)
            img_src = cv2.imread(img_src_path)
            img_dst = cv2.resize(img_src, (img_w, img_h))
            cv2.imwrite(img_dst_path, img_dst)
        # copy the file
        for f in self.lab_root:
            f_src_path = os.path.join(self.img_root, f)
            f_dst_path = os.path.join(new_lab_dir, f)
            open(f_dst_path, 'w').write(open(f_src_path, 'r').read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tool set for building the Oxford dataset')
    parser.add_argument('--root_dir', type=str, required=False, help='root path to the Oxford dataset')
    parser.set_defaults(root_dir='/home/gordonwzhe/data/Oxford')
    args = parser.parse_args()

    oxford_dataset = OxfordDataset(args.root_dir)

    # # make the training set and the test set of the Oxford dataset
    # training_dir = '/home/gordonwzhe/data/Oxford/training/'
    # test_dir = '/home/gordonwzhe/data/Oxford/test/'
    # oxford_dataset.make_training_test_set(training_dir, test_dir, img_size=512)

    # # Calculates each channel's mean value of RGB images in the training set
    # print(oxford_dataset.cal_image_mean_channel())

    # make a dataset with the same images in which the resolution is united
    new_dataset_path = '/home/gordonwzhe/data/Oxford/uni-oxford/'
    oxford_dataset.uni_dataset(new_dataset_path, img_h=384, img_w=512)
