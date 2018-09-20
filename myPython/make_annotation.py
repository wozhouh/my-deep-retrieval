# -*- coding: utf-8 -*-

# Python script that generates the annotation file (image file name + class index)
# Run the make_train_test.py first or you are sure about having the training/test set data already

# usage: python ./myPython/make_annotation.py
#   --S 512
#   --dataset ~/data/myOxford/train-512/
#   --file train-512.txt


import os
import argparse
import random
from class_helper import *
from make_train_test import transform_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make the data into lmdb format')
    parser.add_argument('--S', type=int, required=True, help='Resize larger side of image to S pixels (e.g. 800)')
    parser.add_argument('--dataset', type=str, required=False, help='Path to the Oxford / Paris directory')
    parser.add_argument('--file', type=str, required=False, help='Path to the generated annotation file')
    parser.set_defaults(S=512)
    parser.set_defaults(dataset='/home/processyuan/data/myOxford/train/')
    parser.set_defaults(file='train.txt')
    args = parser.parse_args()

    # Load the dataset
    dataset = Dataset(args.dataset)

    # path to directory
    img_dir = 'jpg'
    lab_dir = 'lab'
    cls_dir = 'cls'
    img_root = os.path.join(args.dataset, img_dir)
    lab_root = os.path.join(args.dataset, lab_dir)
    labels = os.path.join(args.dataset, args.file)
    lines_labels = []

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
            # we choose the 'good' and 'ok' images as positive samples
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
            transform_image(args.S, src_img, dst_img)
            # open(dst_img, 'wb').write(open(src_img, 'rb').read()) # save the origin image
            lines_labels.append(img+' '+str(k)+'\n')

    # write the file 'train.txt'
    random.shuffle(lines_labels)
    if os.path.isfile(labels):
        os.remove(labels)
    f_labels = open(labels, 'w')
    for line in lines_labels:
        f_labels.write(line)

    f_labels.close()
