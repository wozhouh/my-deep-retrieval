# -*- coding: utf-8 -*-

# Python class with common operations on the Paris dataset

import os
import argparse
import cv2
import numpy as np
import random
from region_generator import *


class ParisDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.cls_dir = os.path.join(self.root_dir, 'cls')
        self.lab_dir = os.path.join(self.root_dir, 'lab')
        self.training_dir = os.path.join(self.root_dir, 'training')
        self.test_dir = os.path.join(self.root_dir, 'test')
        self.triplet_dir = os.path.join(self.root_dir, 'triplet')
        self.aug_dir = os.path.join(self.root_dir, 'aug')
        self.blacklist = []
        self.L = 2
        self.mean = np.array([117.80904, 130.27611, 134.65074], dtype=np.float32)[:, None, None]
        self.dataset = []  # list of image names
        self.queries = []  # list of image names
        self.q_fname = []  # list of image names
        self.a_fname = []  # list of image names in list of classes
        self.a_idx = []  # list of image indices in list of classes
        self.num_queries = 0
        self.num_dataset = 0
        f_blacklist = open(os.path.join(self.root_dir, 'paris_blacklist.txt'), 'r')
        lines = f_blacklist.readlines()
        for line in lines:
            self.blacklist.append(line.strip())

        if os.path.exists(self.triplet_dir):
            self.cls_list = os.listdir(self.triplet_dir)
            self.cls_list.remove('useless')

    # test set is copied from 'cls' directory and resized to given shape and images in blacklist are removed
    def make_test_set(self, img_h=None, img_w=None):
        img_dir = os.path.join(self.training_dir, 'img')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        for cls in os.listdir(self.cls_dir):
            cls_path = os.path.join(self.cls_dir, cls)
            for img in os.listdir(cls_path):
                img_src_path = os.path.join(cls_path, img)
                img_dst_path = os.path.join(img_dir, img)
                if img in self.blacklist:
                    os.remove(img_src_path)
                else:
                    # copy the images with original resolution
                    if img_h is None and img_w is None:
                        open(img_dst_path, 'wb').write(open(img_src_path, 'rb').read())
                    # copy the images with given resolution
                    else:
                        cv2.imwrite(img_dst_path, cv2.resize(cv2.imread(img_src_path), (img_w, img_h)))

    # images are classified into 'ok' directory and 'junk' directory and copied from '${ROOT_DIR}/training/img/'
    # Note that there does not exist a txt starting with 'general' so we remove its directory mannully
    def make_triplet_set(self):
        txt_list = os.listdir(self.lab_dir)
        cls_list = os.listdir(self.cls_dir)
        img_dir = os.path.join(self.training_dir, 'img')
        img_list = os.listdir(img_dir)
        img_used_set = set([])
        for cls in cls_list:
            img_cls_dir = os.path.join(self.triplet_dir, cls)
            if os.path.exists(img_cls_dir):
                for f in os.listdir(img_cls_dir):
                    os.remove(os.path.join(img_cls_dir, f))
            else:
                os.makedirs(img_cls_dir)
            os.makedirs(os.path.join(img_cls_dir, 'ok'))
            os.makedirs(os.path.join(img_cls_dir, 'junk'))
            for txt in txt_list:
                if txt.startswith(cls):
                    if txt.endswith('ok.txt'):
                        lines = open(os.path.join(self.lab_dir, txt), 'r')
                        for line in lines:
                            img_name = line.strip() + '.jpg'
                            img_src_path = os.path.join(img_dir, img_name)
                            img_dst_path = os.path.join(img_cls_dir, 'ok', img_name)
                            if not os.path.exists(img_dst_path) and img_name in img_list:
                                open(img_dst_path, 'wb').write(open(img_src_path, 'rb').read())
                                img_used_set.add(img_name)
                    if txt.endswith('junk.txt'):
                        lines = open(os.path.join(self.lab_dir, txt), 'r')
                        for line in lines:
                            img_name = line.strip() + '.jpg'
                            img_src_path = os.path.join(img_dir, img_name)
                            img_dst_path = os.path.join(img_cls_dir, 'junk', img_name)
                            if not os.path.exists(img_dst_path) and img_name in img_list:
                                open(img_dst_path, 'wb').write(open(img_src_path, 'rb').read())
                                img_used_set.add(img_name)

        useless_dir = os.path.join(self.triplet_dir, 'useless')
        if os.path.exists(useless_dir):
            for f in os.listdir(useless_dir):
                os.remove(os.path.join(useless_dir, f))
        else:
            os.makedirs(useless_dir)
        for img in img_list:
            if img not in img_used_set:
                img_src_path = os.path.join(img_dir, img)
                img_dst_path = os.path.join(useless_dir, img)
                if not os.path.exists(img_dst_path) and img in img_list:
                    open(img_dst_path, 'wb').write(open(img_src_path, 'rb').read())

    # perform data-augmentation by randomly resizing (assuming that the '/aug/img/' directory already exists)
    def make_aug_set(self, obj_h=384, obj_w=512, resize_num=2):
        for cls in os.listdir(self.cls_dir):
            cls_path = os.path.join(self.cls_dir, cls)
            for i in os.listdir(cls_path):
                if i not in self.blacklist:
                    img_src_path = os.path.join(cls_path, i)
                    img_dst_path = os.path.join(self.aug_dir, 'img', i)
                    # open(img_dst_path, 'wb').write(open(img_src_path, 'rb').read())
                    img = cv2.imread(img_src_path)
                    img_dst = cv2.resize(img, (obj_w, obj_h))
                    cv2.imwrite(img_dst_path, img_dst)
                    img_shape = img.shape
                    img_h = img_shape[0]
                    img_w = img_shape[1]
                    if obj_h < img_h and obj_w < img_w:
                        # resize (crop to the middle size and then resize)
                        r_obj_h = obj_h + (img_h - obj_h) / 2
                        r_obj_w = obj_w + (img_w - obj_w) / 2
                        r_room_h = img_h - r_obj_h
                        r_room_w = img_w - r_obj_w
                        for k in range(resize_num):
                            r_rand_h = random.randint(0, r_room_h)
                            r_rand_w = random.randint(0, r_room_w)
                            r_img_crop = img[r_rand_h: r_rand_h + r_obj_h, r_rand_w: r_rand_w + r_obj_w, :]
                            img_resize = cv2.resize(r_img_crop, (obj_w, obj_h))
                            cv2.imwrite(img_dst_path.split('.jpg')[0] + '_resize_' + str(k) + '.jpg', img_resize)


    # make the directories of 'queries' and 'dataset' for evaluating the embedding vector
    # images in 'triplet' directory are a little different from the way of 'make_triplet_set' above
    # every 'ok' image selected extracted into a separated directory of their class and others into a 'junk' directory
    # images of 'dataset' are made up of the images randomly sampled from clean 'triplet' directory
    # and the irrelevant images from the 'useless' directory
    # images of 'queries' are manually selected, which is in high quality of the presented building
    def make_dataset_for_test(self, num_cls=30, num_useless=1200):
        dataset_dir = os.path.join(self.test_dir, 'dataset')
        useless_dir = os.path.join(self.triplet_dir, 'junk', 'useless')
        for cls in self.cls_list:
            cls_path = os.path.join(self.triplet_dir, cls)
            img_cls_sampled = random.sample(os.listdir(cls_path), num_cls)
            for k, i in enumerate(img_cls_sampled):
                img_src_path = os.path.join(cls_path, i)
                img_dst_path = os.path.join(dataset_dir, cls+str(k)+'.jpg')
                open(img_dst_path, 'wb').write(open(img_src_path, 'rb').read())
        img_useless_sampled = random.sample(os.listdir(useless_dir), num_useless)
        for k, i in enumerate(img_useless_sampled):
            img_src_path = os.path.join(useless_dir, i)
            img_dst_path = os.path.join(dataset_dir, 'useless' + str(k) + '.jpg')
            open(img_dst_path, 'wb').write(open(img_src_path, 'rb').read())

    # Initializes the list of queries and answers for future use of precision and mAP calculation
    def get_queries_answer_list(self):
        queries_dir = os.path.join(self.test_dir, 'queries')
        dataset_dir = os.path.join(self.test_dir, 'dataset')
        self.dataset = os.listdir(dataset_dir)
        self.queries = os.listdir(queries_dir)
        dataset_dict = {}
        for cls in self.cls_list:
            for d in self.dataset:
                if cls in d:
                    if cls in dataset_dict.keys():
                        dataset_dict[cls].append(d)
                    else:
                        dataset_dict[cls] = [d]
        for cls in self.cls_list:
            for q in self.queries:
                if cls in q:
                    self.q_fname.append(q)
                    self.a_fname.append(dataset_dict[cls])
                    self.a_idx.append([self.dataset.index(i) for i in dataset_dict[cls]])

        self.num_queries = len(self.q_fname)
        self.num_dataset = len(self.dataset)

    def prepare_image_and_grid_regions_for_network(self, img_dir, fname):
        img = self.load_image(os.path.join(self.test_dir, img_dir, fname))
        all_regions = [get_rmac_region_coordinates(img.shape[1], img.shape[2], self.L)]
        regions = pack_regions_for_network(all_regions)
        return np.expand_dims(img, axis=0), regions

    def load_image(self, fname):
        return cv2.imread(fname).transpose(2, 0, 1) - self.mean

    # Calculates the value of mAP according to the standard of VOC2010 and later
    def cal_mAP(self, sim):
        assert len(sim.shape) == 2, 'This is a 2-dim similarity matrix'
        assert sim.shape[0] == self.num_queries, 'number of rows should be equal to number of queries'
        assert sim.shape[1] == self.num_dataset, 'number of columns should be equal to number of dataset'
        q_AP = np.zeros(self.num_queries, dtype=np.float32)
        idx = np.argsort(sim, axis=1)[:, ::-1]
        for q in range(self.num_queries):
            cnt_correct_last = 0
            q_ans = self.a_idx[q]
            recall = np.zeros(len(q_ans), dtype=np.float32)
            for d in range(self.num_dataset):
                top_k = d + 1
                top_idx = list(idx[q, : top_k])
                cnt_correct = len([i for i in top_idx if i in q_ans])
                assert cnt_correct >= cnt_correct_last
                assert cnt_correct <= len(q_ans)
                if cnt_correct > cnt_correct_last:
                    recall[cnt_correct - 1] = float(cnt_correct) / float(top_k)  # precision under the given recall
                    cnt_correct_last = cnt_correct
                if cnt_correct == len(q_ans):
                    break
            # calculates the maximum precision when no less than given recall
            recall_max = np.zeros(len(q_ans), dtype=np.float32)
            for r in range(len(q_ans)):
                recall_max[r] = np.max(recall[r:])
            q_AP[q] = recall_max.mean(axis=0)
            print("AP of query %s: %f" % (self.q_fname[q], q_AP[q]))
        return q_AP.mean(axis=0) * 100.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tool set for building the Paris dataset')
    parser.add_argument('--root_dir', type=str, required=False, help='root path to the Paris dataset')
    parser.set_defaults(root_dir='/home/processyuan/data/Paris/')
    args = parser.parse_args()

    paris_dataset = ParisDataset(args.root_dir)

    # # Generates the test set with a uniform resolution
    # paris_dataset.make_test_set(img_h=384, img_w=512)

    # # Generates the triplet set
    # paris_dataset.make_triplet_set()

    # # Generates the augmentation set
    # paris_dataset.make_aug_set(obj_h=384, obj_w=512, resize_num=2)

    # paris_dataset.get_queries_answer_list()
