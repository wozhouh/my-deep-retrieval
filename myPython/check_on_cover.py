# -*- coding: utf-8 -*-

import argparse
import os


class ValCoverDataset:
    def __init__(self, new_mapped, old_mapped, img_dir, val_dir):
        self.new_mapped = new_mapped
        self.old_mapped = old_mapped
        self.img_dir = img_dir
        self.val_dir = val_dir
        self.new_mapped_lines = open(self.new_mapped, 'r').readlines()
        self.old_mapped_lines = open(self.old_mapped, 'r').readlines()
        if os.path.exists(self.val_dir):
            os.makedirs(self.val_dir)

    def build_val_subset(self, cls_num=300):
        cls_cnt = 0
        anchor_list = []
        new_mapped_dir = os.path.join(self.val_dir, "new")
        old_mapped_dir = os.path.join(self.val_dir, "old")
        os.makedirs(new_mapped_dir)
        os.makedirs(old_mapped_dir)
        for l in self.new_mapped_lines:
            img_num = len(l.split(" "))
            if img_num > 3:
                cls_cnt += 1
                if cls_cnt >= 300:
                    break
                anchor = l.strip().split("\t")[0]
                mapped_item_score_temp = (l.strip().split("\t")[1]).split(" ")
                mapped_item = [t.split(":")[0] for t in mapped_item_score_temp]
                mapped_score = [t.split(":")[1] for t in mapped_item_score_temp]
                anchor_dir = os.path.join(new_mapped_dir, anchor)
                os.makedirs(anchor_dir)
                anchor_list.append(anchor)
                for k, item in enumerate(mapped_item):
                    src_img_name = item + '.jpg'
                    dst_img_name = item + '_' + mapped_score[k] + '.jpg'
                    dst_img_path = os.path.join(anchor_dir, dst_img_name)
                    src_img_path = os.path.join(self.img_dir, src_img_name)
                    open(dst_img_path, 'wb').write(open(src_img_path, 'rb').read())
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build the cover dataset for de-duplicate validation')
    parser.add_argument('--img_dir', type=str, required=True, help='file path to the image directory')
    parser.add_argument('--new_txt', type=str, required=True, help='file path to the file of mapping on the new model')
    parser.add_argument('--old_txt', type=str, required=True, help='file path to the file of mapping on the old model')
    args = parser.parse_args()

    valCoverDataset = ValCoverDataset(args.new_txt, args.old_txt, args.img_dir, args.val_dir)