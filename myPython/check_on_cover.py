# -*- coding: utf-8 -*-

# Python script that validates the embedding model for duplicate removal on the cover dataset
# similarity scores pre-calculated and images belonging to a group are written into a line of the mapped file

""" example usage:
    python check_on_cover.py \
        --img_dir D:\\Data\\cover\\full\\img \
        --new_txt D:\\Data\\cover\\full\\mapped\\newsimfea512_0.65_res_mapped.txt \
        --old_txt D:\\Data\\cover\\full\\mapped\\oldmodel_res_mapped.txt
"""

import argparse
import os
import random


class ValCoverDataset:
    def __init__(self, new_mapped_file, old_mapped_file, img_dir):
        self.new_mapped_file = new_mapped_file
        self.old_mapped_file = old_mapped_file
        self.img_dir = img_dir
        self.anchor_list = []

    # build a new anchor list as queries and save the images of the same group under a directory
    def make_anchor_list_and_save_groups(self, mapped_file, group_dir, th=0., min_img_num_per_group=5, run_groups=None):
        cls_cnt = 0  # counts how many groups processed
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)
        lines = open(mapped_file, 'r').readlines()
        for l in lines:
            l_split = l.strip().split("\t")
            anchor = l_split[0]
            mapped_item_score_temp = (l_split[1]).split(" ")
            mapped_score = []
            mapped_item = []
            for t in mapped_item_score_temp:
                score_str = t.split(":")[-1]
                if float(score_str) >= th:
                    mapped_score.append(score_str)
                    mapped_item.append(t.split(":")[0])
                if len(mapped_score) <= min_img_num_per_group:
                    continue
                cls_cnt += 1
                if run_groups is not None and cls_cnt > run_groups:
                    break

                anchor_dir = os.path.join(group_dir, anchor)
                if not os.path.exists(anchor_dir):
                    os.makedirs(anchor_dir)

                self.anchor_list.append(anchor)
                for k, item in enumerate(mapped_item):
                    src_img_name = item + '.jpg'
                    dst_img_name = item + '_' + mapped_score[k] + '.jpg'
                    dst_img_path = os.path.join(anchor_dir, dst_img_name)
                    src_img_path = os.path.join(self.img_dir, src_img_name)
                    open(dst_img_path, 'wb').write(open(src_img_path, 'rb').read())

    # save the images of the same group under a directory according to the given anchor list
    def save_groups_from_anchor_list(self, mapped_file, group_dir, th=0.):
        lines = open(mapped_file, 'r').readlines()
        if len(self.anchor_list) == 0:
            print("WARNING: anchor list empty")
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)

        # find which line the queries are in (key: image name; value: line index)
        image_line_idx = {}
        for k, l in enumerate(lines):
            anchor = l.strip().split("\t")[0]
            if anchor in self.anchor_list:
                image_line_idx[anchor] = k

        for a in self.anchor_list:
            line = lines[image_line_idx[a]]
            mapped_item_score_temp = (line.strip().split("\t")[1]).split(" ")
            mapped_score = []
            mapped_item = []
            for t in mapped_item_score_temp:
                score_str = t.split(":")[-1]
                if float(score_str) >= th:
                    mapped_score.append(score_str)
                    mapped_item.append(t.split(":")[0])

            anchor_dir = os.path.join(group_dir, a)
            if not os.path.exists(anchor_dir):
                os.makedirs(anchor_dir)

            for k, item in enumerate(mapped_item):
                src_img_name = item + '.jpg'
                dst_img_name = item + '_' + mapped_score[k] + '.jpg'
                dst_img_path = os.path.join(anchor_dir, dst_img_name)
                src_img_path = os.path.join(self.img_dir, src_img_name)
                open(dst_img_path, 'wb').write(open(src_img_path, 'rb').read())

    # raise similarity threshold for filtering and count how many images left (build the directory of groups first)
    def check_threshold(self, group_dir, th):
        cnt = 0
        for q in os.listdir(group_dir):
            q_path = os.path.join(group_dir, q)
            for img in os.listdir(q_path):
                img_name = img.split('.jpg')[0]
                img_score = float(img_name.split('_')[1])
                if img_score >= th:
                    cnt += 1
        print('number of left images after raising the threshold: %d' % cnt)

    # randomly select n groups and copy them into a directory for later manual annotation
    def select_random_n_groups(self, group_dir, new_dir, n=1000):
        group_all = os.listdir(group_dir)
        group_sample = random.sample(group_all, n)
        for g in group_sample:
            src_group_path = os.path.join(group_dir, g)
            dst_group_path = os.path.join(new_dir, g)
            os.makedirs(dst_group_path)
            for img in os.listdir(src_group_path):
                src_img_path = os.path.join(src_group_path, img)
                dst_img_path = os.path.join(dst_group_path, img)
                open(dst_img_path, 'wb').write(open(src_img_path, 'rb').read())

    # copy the image groups into a directory which are in the list of queries written in 'queries_file'
    def copy_groups_in_queries(self, queries_file, src_dir, dst_dir):
        lines = open(queries_file, 'r')
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for q in lines:
            q = q.strip()
            src_group_path = os.path.join(src_dir, q)
            dst_group_path = os.path.join(dst_dir, q)
            os.makedirs(dst_group_path)
            for img in os.listdir(src_group_path):
                src_img_path = os.path.join(src_group_path, img)
                dst_img_path = os.path.join(dst_group_path, img)
                open(dst_img_path, 'wb').write(open(src_img_path, 'rb').read())

    # find the image groups in the list of queries from the mapped file and copy them into a directory
    def save_groups_in_queries(self, queries_file, dst_dir, th=0.):
        q_lines = open(queries_file, 'r').readlines()
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        self.anchor_list = [l.strip() for l in q_lines]
        self.save_groups_from_anchor_list(self.new_mapped_file, dst_dir, th=th)

    # change the similarity threshold given by the list
    # and see how many same-origin images recalled with every threshold
    def count_img_num_in_mapped_file(self, mapped_file, queries_list, th_list):
        m_lines = open(mapped_file, 'r').readlines()

        # init a dictionary to save the number of good cases found
        th_dict = {}

        # find which line the queries are in (key: image name; value: line index)
        image_line_idx = {}
        for k, l in enumerate(m_lines):
            anchor = l.strip().split("\t")[0]
            image_line_idx[anchor] = k

        for th in th_list:
            cnt = 0
            for q in queries_list:
                line = m_lines[image_line_idx[q]]
                mapped_item_score_temp = (line.strip().split("\t")[1]).split(" ")
                mapped_score = []
                for t in mapped_item_score_temp:
                    score_str = t.split(":")[-1]
                    if float(score_str) >= th:
                        mapped_score.append(score_str)

                line_cnt = len(mapped_score)
                cnt += line_cnt

            th_dict[th] = cnt

        return th_dict

    # count the number of images of the same-origin groups with a threshold between 'low_th' and 'high_th'
    def count_good_img_num(self, queries_file, mapped_file, low_th, high_th):
        q_lines = open(queries_file, 'r').readlines()
        q_list = [l.strip() for l in q_lines]

        # list the similarity threshold for test
        th_list = []
        th_temp = low_th
        while th_temp <= high_th:
            th_list.append(th_temp)
            th_temp += 0.01

        good_img_count_dict = self.count_img_num_in_mapped_file(mapped_file=mapped_file, queries_list=q_list,
                                                                th_list=th_list)
        print(good_img_count_dict)

    # count the number of images of the irrelevant groups with a threshold between 'low_th' and 'high_th'
    def count_junk_img_num(self, queries_file, mapped_file, src_group_path, low_th, high_th):
        q_lines = open(queries_file, 'r').readlines()
        q_list = [l.strip() for l in q_lines]
        junk_list = []

        # make the list of queries for junk groups
        group_list = os.listdir(src_group_path)
        for g in group_list:
            if g not in q_list:
                junk_list.append(g)

        # list the similarity threshold for test
        th_list = []
        th_temp = low_th
        while th_temp <= high_th:
            th_list.append(th_temp)
            th_temp += 0.01

        junk_img_count_dict = self.count_img_num_in_mapped_file(mapped_file=mapped_file, queries_list=junk_list,
                                                                th_list=th_list)
        print(junk_img_count_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build the cover dataset for de-duplicate validation')
    parser.add_argument('--img_dir', type=str, required=False, help='path to the image directory')
    parser.add_argument('--new_txt', type=str, required=False, help='file path to the file of mapping on the new model')
    parser.add_argument('--old_txt', type=str, required=False, help='file path to the file of mapping on the old model')
    parser.add_argument('--th', type=float, required=False, help='similarity threshold for selecting images into groups')
    parser.add_argument('--src_dir', type=str, required=False, help='path to the old directory of image groups')
    parser.add_argument('--dst_dir', type=str, required=False, help='path to the new directory of image groups')
    parser.add_argument('--queries_file', type=str, required=False, help='path to the file with list of queries written')
    args = parser.parse_args()

    valCoverDataset = ValCoverDataset(args.new_txt, args.old_txt, args.img_dir)

    # valCoverDataset.make_anchor_list_and_save_groups(valCoverDataset.new_mapped_file, args.dst_dir)

    # valCoverDataset.select_random_n_groups(args.group_dir, args.new_dir)

    # valCoverDataset.copy_groups_in_queries(args.queries_file, args.src_dir, args.dst_dir)

    # valCoverDataset.save_groups_in_queries(args.queries_file, args.dst_dir, args.th)

    # valCoverDataset.count_good_img_num(args.queries_file, args.new_txt, 0.65, 0.75)

    # valCoverDataset.count_junk_img_num(args.queries_file, args.new_txt, args.src_dir, 0.56, 0.70)
