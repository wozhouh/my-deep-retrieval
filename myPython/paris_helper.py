# -*- coding: utf-8 -*-

# Python class with common operations on the Paris dataset

import os
import argparse
import numpy as np
import cv2

def get_rmac_region_coordinates(H, W, L):
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


def pack_regions_for_network(all_regions):
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

class ParisDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.cls_dir = os.path.join(self.root_dir, 'cls')
        self.lab_dir = os.path.join(self.root_dir, 'lab')
        self.test_dir = os.path.join(self.root_dir, 'training')
        self.blacklist = []
        f_blacklist = open(os.path.join(self.root_dir, 'paris_blacklist.txt'), 'r')
        lines = f_blacklist.readlines()
        for line in lines:
            self.blacklist.append(line.strip())

    def make_test_set(self, img_h=384, img_w=512):
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
                    # open(img_dst_path, 'wb').write(open(img_src_path, 'rb').read())
                    cv2.imwrite(img_dst_path, cv2.resize(cv2.imread(img_src_path), (img_w, img_h)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tool set for building the Paris dataset')
    parser.add_argument('--root_dir', type=str, required=False, help='root path to the Paris dataset')
    parser.set_defaults(root_dir='/home/processyuan/data/Paris/')
    args = parser.parse_args()

    paris_dataset = ParisDataset(args.root_dir)
    paris_dataset.make_test_set(img_h=384, img_w=512)

    # cnt = 0
    # img_dir = os.path.join(paris_dataset.test_dir, 'img')
    # for f in os.listdir(img_dir):
    #     img = cv2.imread(os.path.join(img_dir, f))
    #     all_regions = [get_rmac_region_coordinates(img.shape[0], img.shape[1], 2)]
    #     regions = pack_regions_for_network(all_regions)
    #     print(cnt)
    #     print(img.shape)
    #     print(regions.shape)
    #     cnt += 1
