# -*- coding: utf-8 -*-

import caffe
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


class NormalizeLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'
        self.eps = 1e-8

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[:] = bottom[0].data / np.expand_dims(
            self.eps + np.sqrt((bottom[0].data ** 2).sum(axis=1)), axis=1)

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            raise NotImplementedError(
                "Backward pass not supported with this implementation")
        else:
            pass


class AggregateLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'

    def reshape(self, bottom, top):
        tmp_shape = list(bottom[0].data.shape)
        tmp_shape[0] = 1
        top[0].reshape(*tmp_shape)

    def forward(self, bottom, top):
        top[0].data[:] = bottom[0].data.sum(axis=0)

    def backward(self, top, propagate_down, bottom):
        """Get top diff and compute diff in bottom."""
        if propagate_down[0]:
            num = bottom[0].data.shape[0]
            for k in range(num):
                bottom[0].diff[k] = top[0].diff[0]
        # raise NotImplementedError(
        #     "Backward pass not supported with this implementation")


class RigidGridLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.num_region = 8  # typical number of rigid regions is 8 so fix it here
        self.dim_rois = 5  # (index, xmin, ymin, xmax, ymax)
        assert len(bottom) == 1, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'
        assert bottom[0].data.shape[0] == 1, 'Batch size is fixed to 1 as the size of images might be different'
        assert bottom[0].data.shape[1] == 3, 'The input should be a 3-channel RGB image in 1x3xHxW format'

    def reshape(self, bottom, top):
        tmp_shape = [self.num_region, self.dim_rois]
        top[0].reshape(*tmp_shape)

    def forward(self, bottom, top):
        H = bottom[0].data.shape[2]
        W = bottom[0].data.shape[3]
        all_regions = [get_rmac_region_coordinates(H, W, 2)]
        R = pack_regions_for_network(all_regions)
        rois = R.shape[0]
        # if number of regions is less than fixed value, pad it with used regions
        if rois < self.num_region:
            ind = 0
            diff = self.num_region - rois
            top_data = np.zeros((self.num_region, self.dim_rois), dtype=np.float32)
            top_data[: rois, :] = R
            for k in range(diff):
                top_data[rois+k, :] = R[ind, :]
                ind += 1
                if ind == rois:
                    ind = 0
            top[0].data[:] = top_data
        else:
            top[0].data[:] = R[: self.num_region, :]

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            raise NotImplementedError(
                "Backward pass not supported with this implementation")
        else:
            pass
