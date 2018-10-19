# -*- coding: utf-8 -*-

# Python function that generates rigid grid (roi) given the shape of image
# usage: all_regions = [get_rmac_region_coordinates(img_h, img_w, L)]
#        R = pack_regions_for_network(all_regions)

'''
Note:
    The generation of rigid grid is from the original test.py, which make the intersection of neighboring grids
    less than 0.4. But mostly the shape of images we process will not change, so sometimes we assign the grid
    directly. For example, for cover images of shape (280, 496, 3), the grids are as follow:
    [[  0.   0.   0. 279. 279.]
    [  0. 216.   0. 495. 279.]
    [  0.   0.   0. 185. 185.]
    [  0. 155.   0. 340. 185.]
    [  0. 310.   0. 495. 185.]
    [  0.   0.  94. 185. 279.]
    [  0. 155.  94. 340. 279.]
    [  0. 310.  94. 495. 279.]
    for Oxford/Paris dataset used to train, the image shape is (384, 512, 3) and the grids are
    [[0.   0.   0. 383. 383.]
    [0. 128.   0. 511. 383.]
    [0.   0.   0. 255. 255.]
    [0. 128.   0. 383. 255.]
    [0. 256.   0. 511. 255.]
    [0.  0.  128. 255. 383.]
    [0. 128. 128. 383. 383.]
    [0. 256. 128. 511. 383.]]
'''


import numpy as np

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