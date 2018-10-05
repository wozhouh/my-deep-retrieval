# -*- coding: utf-8 -*-

# Python class with common operations on the Paris dataset

import os
import argparse
import cv2
from region_generator import  *


class ParisDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.cls_dir = os.path.join(self.root_dir, 'cls')
        self.lab_dir = os.path.join(self.root_dir, 'lab')
        self.test_dir = os.path.join(self.root_dir, 'training')
        self.triplet_dir = os.path.join(self.root_dir, 'triplet')
        self.blacklist = []
        f_blacklist = open(os.path.join(self.root_dir, 'paris_blacklist.txt'), 'r')
        lines = f_blacklist.readlines()
        for line in lines:
            self.blacklist.append(line.strip())

    # test set is copied from 'cls' directory and resized to given shape and images in blacklist are removed
    def make_test_set(self, img_h=None, img_w=None):
        img_dir = os.path.join(self.test_dir, 'img')
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
        img_dir = os.path.join(self.test_dir, 'img')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tool set for building the Paris dataset')
    parser.add_argument('--root_dir', type=str, required=False, help='root path to the Paris dataset')
    parser.set_defaults(root_dir='/home/processyuan/data/Paris/')
    args = parser.parse_args()

    paris_dataset = ParisDataset(args.root_dir)

    # # Generates the test set with a uniform resolution
    # paris_dataset.make_test_set(img_h=384, img_w=512)

    # Generates the triplet set
    paris_dataset.make_triplet_set()

