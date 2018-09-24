# -*- coding: utf-8 -*-

import caffe
from CoverData import *


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
        # assert bottom[0].data.shape[0] == 1, 'Batch size is fixed to 1 as the size of images might be different'
        assert bottom[0].data.shape[1] == 3, 'The input should be a 3-channel RGB image in 1x3xHxW format'

    def reshape(self, bottom, top):
        tmp_shape = [self.num_region, self.dim_rois]
        top[0].reshape(*tmp_shape)

    def forward(self, bottom, top):
        H = bottom[0].data.shape[2]
        W = bottom[0].data.shape[3]
        all_regions = [get_rmac_region_coordinates(H, W, 2)]
        R = pack_regions_for_network(all_regions)

        batch_size = bottom[0].data.shape[0]
        reg = np.zeros((self.num_region * batch_size, self.dim_rois), dtype=np.float32)
        for k in range(batch_size):
            R_temp = R[:]
            reg[k*self.num_region: (k+1)*self.num_region, :] =

        top[0].data[:] = R[: self.num_region, :]
        # rois = R.shape[0]
        # # if number of regions is less than fixed value, pad it with used regions
        # if rois < self.num_region:
        #     ind = 0
        #     diff = self.num_region - rois
        #     top_data = np.zeros((self.num_region, self.dim_rois), dtype=np.float32)
        #     top_data[: rois, :] = R
        #     for k in range(diff):
        #         top_data[rois+k, :] = R[ind, :]
        #         ind += 1
        #         if ind == rois:
        #             ind = 0
        #     top[0].data[:] = top_data
        # else:
        #     top[0].data[:] = R[: self.num_region, :]

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            raise NotImplementedError(
                "Backward pass not supported with this implementation")
        else:
            pass


    class ResizeLayer(caffe.Layer):
        def setup(self, bottom, top):
            assert len(bottom) == 1, 'This layer can only have one bottom'

        def reshape(self, bottom, top):
