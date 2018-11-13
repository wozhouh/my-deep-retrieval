# -*- coding: utf-8 -*-

# Python class with common operations on the Landmark dataset

import argparse
import os
import cv2
import random
# from urllib.request import urlretrieve  # used in Python 3, annotated first


class LandMarkDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.csv_dir = os.path.join(self.root_dir, 'csv')
        self.csv_file = os.path.join(self.csv_dir, 'train.csv')
        self.raw_dir = os.path.join(self.root_dir, 'raw')
        self.cls_dir = os.path.join(self.root_dir, 'cls')
        self.training_dir = os.path.join(self.root_dir, 'training')
        self.url_dict = {}
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
            os.makedirs(os.path.join(self.raw_dir, 'cls'))
        if not os.path.exists(self.cls_dir):
            os.makedirs(self.cls_dir)
        if not os.path.exists(self.training_dir):
            os.makedirs(self.training_dir)

    # get the image url of each landmark which has image between l and h
    def read_url_from_file(self, l, h):
        url_temp = {}
        f = open(self.csv_file, 'r')
        lines = f.readlines()
        lines = lines[1:]  # give up the first line
        for line in lines:
            line_apart = line.strip().split(',')
            if len(line_apart) != 3 or line_apart[2] == 'None':
                continue
            landmark_id = int(line_apart[2])
            if landmark_id not in url_temp.keys():
                url_temp[landmark_id] = [line_apart[1].strip('"')]
            else:
                url_temp[landmark_id].append(line_apart[1].strip('"'))
        # print(self.url_dict)
        img_cnt = 0
        keys = url_temp.keys()
        for k in keys:
            if l <= len(url_temp[k]) <= h:
                self.url_dict[k] = url_temp[k]
                img_cnt += len(self.url_dict[k])
        print('Total number of landmarks selected: %d' % len(self.url_dict.keys()))
        print('Total number of images selected: %d' % img_cnt)

    # download the images from the url (run read_url_from_file() first)
    # skip the images whose urls are not accessible
    def download_image(self):
        raw_cls_dir = os.path.join(self.raw_dir, 'raw-cls')
        broken_url = []
        downloaded_id = []
        broken_log = os.path.join(self.raw_dir, 'broken_url.log')
        downloaded_log = os.path.join(self.raw_dir, 'downloaded_id.log')
        # check which ids have been downloaded
        if os.path.exists(downloaded_log):
            r_downloaded_log = open(downloaded_log, 'r')
            for line in r_downloaded_log.readlines():
                downloaded_id.append(int(line.strip()))
            r_downloaded_log.close()
        # check which urls are broken
        if os.path.exists(broken_log):
            r_broken_log = open(broken_log, 'r')
            for line in r_broken_log.readlines():
                broken_url.append(line.strip())
            r_broken_log.close()

        w_downloaded_log = open(downloaded_log, 'a')
        w_broken_log = open(broken_log, 'a')

        for landmark_id in self.url_dict.keys():
            if landmark_id in downloaded_id:
                print("landmark %s downloaded" % landmark_id)
                continue
            cls_path = os.path.join(raw_cls_dir, str(landmark_id))
            if not os.path.exists(cls_path):
                os.makedirs(cls_path)
            for k, url in enumerate(self.url_dict[landmark_id]):
                img_path = os.path.join(cls_path, str(k)+'.jpg')
                if not os.path.exists(img_path):
                    # pass if the url is broken
                    if url not in broken_url:
                        try:
                            urlretrieve(url, img_path)
                        except Exception as e:
                            # if downloading fails, add the url to the black-list
                            w_broken_log.write(url + '\n')
                            print('image of url %s missing ...' % url)

            w_downloaded_log.write(str(landmark_id)+'\n')
            print("%d images of landmark %d downloaded" % (len(self.url_dict[landmark_id]), landmark_id))

        w_downloaded_log.close()
        w_broken_log.close()

    # simply clean the dataset by deleting the images whose resolution is lower than the given and crop/resize the rest
    def unite_images_size(self, img_h=288, img_w=384, least_img_per_cls=3, method="crop"):
        img_dst_ratio = float(img_w) / float(img_h)
        raw_cls_dir = os.path.join(self.raw_dir, 'raw-cls')
        useless_cnt = 0
        useful_cnt = 0
        for c in os.listdir(raw_cls_dir):
            raw_cls_path = os.path.join(raw_cls_dir, c)
            cls_path = os.path.join(self.cls_dir, c)
            img_num = len(os.listdir(raw_cls_path))
            if img_num < least_img_per_cls:
                continue
            if not os.path.exists(cls_path):
                os.makedirs(cls_path)
            for i in os.listdir(raw_cls_path):
                img_src_path = os.path.join(raw_cls_path, i)
                img_dst_path = os.path.join(cls_path, i)
                img = cv2.imread(img_src_path)
                if img is None:
                    os.remove(img_src_path)
                    continue
                else:
                    if method == "resize":
                        if img.shape[1] >= img.shape[0] >= img_h and img.shape[1] >= img_w:
                            useful_cnt += 1
                            print("processing the %d image" % useful_cnt)
                            img_resized = cv2.resize(img, (img_w, img_h))
                            cv2.imwrite(img_dst_path, img_resized)
                    else:
                        # give up the images which are too small
                        if img.shape[0] >= img_h and img.shape[1] >= img_w:
                            useful_cnt += 1
                            print("processing the %d image" % useful_cnt)
                            img_src_ratio = float(img.shape[1]) / float(img.shape[0])
                            # too wide
                            if img_src_ratio > img_dst_ratio:
                                img_src_h = img_h
                                img_src_w = int(float(img_src_h) * img_src_ratio)
                                img_resized = cv2.resize(img, (img_src_w, img_src_h))
                                # crop the center part on the axis of width
                                img_cropped = img_resized[:, img_src_w/2-img_w/2: img_src_w/2+img_w/2, :]
                            # too high
                            else:
                                img_src_w = img_w
                                img_src_h = int(float(img_src_w) / img_src_ratio)
                                img_resized = cv2.resize(img, (img_src_w, img_src_h))
                                # crop the center part on the axis of height
                                img_cropped = img_resized[img_src_h/2-img_h/2: img_src_h/2+img_h/2, :, :]
                            cv2.imwrite(img_dst_path, img_cropped)
                        else:
                            useless_cnt += 1
        print("%d images finished, %d images deprecated" % (useful_cnt, useless_cnt))

    # revised from "unite_images_size(...) in order to build a training set in which the images are of the same size"
    def make_training_set(self, img_h=384, img_w=512):
        img_dst_ratio = float(img_w) / float(img_h)
        raw_cls_dir = os.path.join(self.raw_dir, 'raw-cls')
        useless_cnt = 0
        useful_cnt = 0
        for c in os.listdir(raw_cls_dir):
            raw_cls_path = os.path.join(raw_cls_dir, c)
            for i in os.listdir(raw_cls_path):
                img_src_path = os.path.join(raw_cls_path, i)
                img_dst_path = os.path.join(self.training_dir, str(useful_cnt)+'.jpg')
                img = cv2.imread(img_src_path)
                if img is None:
                    os.remove(img_src_path)
                    continue
                else:
                    # give up the images which are too small
                    if img.shape[0] >= img_h and img.shape[1] >= img_w:
                        useful_cnt += 1
                        print("processing the %d image" % useful_cnt)
                        img_src_ratio = float(img.shape[1]) / float(img.shape[0])
                        # too wide
                        if img_src_ratio > img_dst_ratio:
                            img_src_h = img_h
                            img_src_w = int(float(img_src_h) * img_src_ratio)
                            img_resized = cv2.resize(img, (img_src_w, img_src_h))
                            # crop the center part on the axis of width
                            img_cropped = img_resized[:, img_src_w / 2 - img_w / 2: img_src_w / 2 + img_w / 2, :]
                        # too high
                        else:
                            img_src_w = img_w
                            img_src_h = int(float(img_src_w) / img_src_ratio)
                            img_resized = cv2.resize(img, (img_src_w, img_src_h))
                            # crop the center part on the axis of height
                            img_cropped = img_resized[img_src_h / 2 - img_h / 2: img_src_h / 2 + img_h / 2, :, :]
                        cv2.imwrite(img_dst_path, img_cropped)
                    else:
                        useless_cnt += 1
        print("%d images finished, %d images deprecated" % (useful_cnt, useless_cnt))

    # count how many images are there in every class
    def count_img_per_class(self):
        cnt_dict = {}
        img_cnt = 0
        for c in os.listdir(self.cls_dir):
            cls_path = os.path.join(self.cls_dir, c)
            img_num = len(os.listdir(cls_path))
            if img_num in cnt_dict.keys():
                cnt_dict[img_num] = cnt_dict[img_num] + 1
            else:
                cnt_dict[img_num] = 1
        for k in cnt_dict.keys():
            img_cnt += k * cnt_dict[k]
        print("Total number of images: %d" % img_cnt)
        print(cnt_dict)

    # write an annotation file for making lmdb
    def make_landmark_cls_annotations(self):
        cls_txt = os.path.join(self.root_dir, 'cls.txt')
        f = open(cls_txt, 'w')
        lines = []
        for c in os.listdir(self.cls_dir):
            cls_path = os.path.join(self.cls_dir, c)
            for i in os.listdir(cls_path):
                img_path = os.path.join(c, i)
                line = img_path + ' ' + c + '\n'
                lines.append(line)
        random.shuffle(lines)
        for line in lines:
            f.write(line)
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build the landmark dataset')
    parser.add_argument('--root_dir', type=str, required=False, help='root path to the Paris dataset')
    parser.set_defaults(root_dir='/home/processyuan/data/Landmark')
    args = parser.parse_args()

    landmark_dataset = LandMarkDataset(args.root_dir)

    # # download the images from urls (used on Windows)
    # landmark_dataset.read_url_from_file(l=100, h=300)
    # landmark_dataset.download_image()

    # landmark_dataset.unite_images_size()
    # landmark_dataset.count_img_per_class()
    # landmark_dataset.make_landmark_cls_annotations()

    landmark_dataset.make_training_set()
