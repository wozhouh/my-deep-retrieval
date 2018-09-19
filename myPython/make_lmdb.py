# -*- coding: utf-8 -*-

# usage: python make_lmdb.py --dataset ...
#     then run the command:

import os
import argparse
import random
from class_helper import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make the data into lmdb format')
    parser.add_argument('--dataset', type=str, required=False, help='Path to the Oxford / Paris directory')
    parser.set_defaults(dataset='/home/processyuan/data/myOxford/train/')
    args = parser.parse_args()

    # Load the dataset
    dataset = Dataset(args.dataset)

    # path to directory
    img_dir = 'jpg'
    lab_dir = 'lab'
    cls_dir = 'cls'
    img_root = os.path.join(args.dataset, img_dir)
    lab_root = os.path.join(args.dataset, lab_dir)
    train_txt = os.path.join(args.dataset, 'train.txt')  # Remeber to change when deployed to test set
    # f_train_txt = open(train_txt, 'r')
    lines_train_txt = []

    # name of every class (11 in total)
    q_names = dataset.q_names
    cls_names = []
    cls_filenames = {}
    for name in q_names:
        cls = name[: -2]
        if cls not in cls_names:
            cls_names.append(cls)

    # make the directory of every class
    for k, cls in enumerate(cls_names):
        cls_temp = []
        for txt in os.listdir(lab_root):
            if txt.startswith(cls) and (txt.endswith('_good.txt') or txt.endswith('_ok.txt')):
                f = open(os.path.join(lab_root, txt), 'r')
                lines = f.readlines()
                for line in lines:
                    cls_temp.append(line.strip()+'.jpg')
        cls_filenames[cls] = cls_temp
        cls_root = os.path.join(args.dataset, cls_dir, cls)

        # delete if exists
        if not os.path.exists(cls_root):
            os.makedirs(cls_root)
        else:
            for f in os.listdir(cls_root):
                full_name = os.path.join(cls_root, f)
                if os.path.isfile(full_name):
                    os.remove(full_name)
        # copy the images
        for img in set(cls_temp):
            src_img = os.path.join(img_root, img)
            dst_img = os.path.join(cls_root, img)
            open(dst_img, 'wb').write(open(src_img, 'rb').read())
            lines_train_txt.append(img+' '+str(k)+'\n')

    # write the file 'train.txt'
    random.shuffle(lines_train_txt)
    if os.path.isfile(train_txt):
        os.remove(train_txt)
    f_train_txt = open(train_txt, 'w')
    for line in lines_train_txt:
        f_train_txt.write(line)

    f_train_txt.close()
