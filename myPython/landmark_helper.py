# -*- coding: utf-8 -*-

# Python class with common operations on the landmark dataset

# import download_helper  # used in Python 3, annotated first
import argparse
import os
import random
import cv2


class LandMarkDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.csv_dir = os.path.join(self.root_dir, 'csv')
        self.csv_file = os.path.join(self.csv_dir, 'training.csv')
        self.training_dir = os.path.join(self.root_dir, 'training')
        self.cls_dir = os.path.join(self.training_dir, 'cls')
        self.url_dict = {}
        if not os.path.exists(self.training_dir):
            os.makedirs(self.training_dir)

    def read_url_from_file(self):
        f = open(self.csv_file, 'r')
        lines = f.readlines()
        for line in lines:
            line_apart = line.strip().split(',')
            if len(line_apart) != 3:
                continue
            building_id = int(line_apart[2])
            if building_id not in self.url_dict.keys():
                self.url_dict[building_id] = [line_apart[1].strip('"')]
            else:
                self.url_dict[building_id].append(line_apart[1].strip('"'))
            print(self.url_dict)
            print(len(self.url_dict.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build the landmark dataset')
    parser.add_argument('--root_dir', type=str, required=False, help='root path to the Paris dataset')
    parser.set_defaults(root_dir='/home/gordonwzhe/data/Paris/')
    args = parser.parse_args()

    landmark_dataset = LandMarkDataset(args.root_dir)
    landmark_dataset.read_url_from_file()
