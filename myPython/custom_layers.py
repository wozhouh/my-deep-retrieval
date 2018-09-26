# -*- coding: utf-8 -*-

import caffe
import numpy as np
import cv2
import yaml
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

# make some changes to the rmac layer so that it can input batch size > 1
class AggregateLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'
        params = yaml.load(self.param_str_)
        self.batch_size = params['batch_size']
        self.num_rois = params['num_rois']
        assert bottom[0].data.shape[0] == self.batch_size * self.num_rois, 'batch_size * num_rois != num_regions'

    def reshape(self, bottom, top):
        tmp_shape = list(bottom[0].data.shape)
        tmp_shape[0] = self.batch_size
        top[0].reshape(*tmp_shape)

    def forward(self, bottom, top):
        # top[0].data[:] = bottom[0].data.sum(axis=0)
        for k in range(self.batch_size):
            bottom_data = bottom[0].data[k * self.num_rois: (k+1) * self.num_rois, ...]
            top[0].data[k, ...] = bottom_data.sum(axis=0)

    def backward(self, top, propagate_down, bottom):
        """Get top diff and compute diff in bottom."""
        # if propagate_down[0]:
        #     num = bottom[0].data.shape[0]
        #     for k in range(num):
        #         bottom[0].diff[k] = top[0].diff[0]
        if propagate_down[0]:
            for k in range(self.batch_size):
                for j in range(self.num_rois):
                    bottom[0].diff[k * self.num_rois + j] = top[0].diff[k]


class LabelLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'
        params = yaml.load(self.param_str_)
        self.dim = params['dim']
        self.features_file = params['features']
        self.batch_size = bottom[0].shape[0]
        self.features = np.load(self.features_file)
        self.batch_features = np.zeros((self.batch_size, self.dim, 1, 1), dtype=np.float32)

    def reshape(self, bottom, top):
        top[0].reshape(*[self.batch_size, self.dim, 1, 1])

    def forward(self, bottom, top):
        feature_idx = np.squeeze(bottom[0].data)
        for k in range(self.batch_size):
            self.batch_features[k] = (self.features[int(feature_idx[k]), :]).reshape(self.dim, 1, 1)
        top[0].data[...] = self.batch_features

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            raise NotImplementedError(
                "Backward pass not supported with this implementation")
        else:
            pass

class RigidGridLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'
        # assert bottom[0].data.shape[0] == 1, 'Batch size is fixed to 1 as the size of images might be different'
        assert bottom[0].data.shape[1] == 3, 'The input should be a 3-channel RGB image in batch x 3 x H x W format'
        self.num_region = 8  # typical number of rigid regions is 8 so fix it here
        self.dim_rois = 5  # (index, xmin, ymin, xmax, ymax)
        self.img_h = bottom[0].data.shape[2]  # for the cover images, h = 280
        self.img_w = bottom[0].data.shape[3]  # for the cover images, w = 496
        self.batch_size = bottom[0].data.shape[0]  # bottom: (batch_size, channel(3), h(280), w(496))

    def reshape(self, bottom, top):
        top[0].reshape(*[self.batch_size * self.num_region, self.dim_rois])  # for the cover images, rois.shape = [11, 5], fix it here

    def forward(self, bottom, top):
        all_regions = [get_rmac_region_coordinates(self.img_h, self.img_w, 2)]
        R = pack_regions_for_network(all_regions)  # for the cover images, R,shape = [8, 5]
        '''
        (1, 3, 280, 496)
        [[  0.   0.   0. 279. 279.]
         [  0. 216.   0. 495. 279.]
         [  0.   0.   0. 185. 185.]
         [  0. 155.   0. 340. 185.]
         [  0. 310.   0. 495. 185.]
         [  0.   0.  94. 185. 279.]
         [  0. 155.  94. 340. 279.]
         [  0. 310.  94. 495. 279.]
        '''
        if self.batch_size == 1:
            top[0].data[:] = R[: self.num_region, :]
        else:
            reg = np.zeros((self.num_region * self.batch_size, self.dim_rois), dtype=np.float32)
            for k in range(self.batch_size):
                R_temp = R
                R_temp[:, 0] = k  # image index in the batch
                reg[k * self.num_region: (k + 1) * self.num_region, :] = R_temp
            top[0].data[:] = reg

        # # if number of regions is less than fixed value, pad it with used regions
        # # if greater, drop the rest
        # rois = R.shape[0]
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
        params = yaml.load(self.param_str_)
        self.s = params['s']
        mean_list = params['mean']
        # param = eval(self.param_str_)
        # self.s = param.get('s', 496)  # what size to resize
        # mean_list = param.get('mean', None)
        self.mean = np.array(mean_list, dtype=np.float32)[:, None, None]
        self.img_size = np.array(bottom[0].data.shape[2: 4])  # for the cover images, h = 280, w = 496
        self.ratio = float(self.s) / np.max(self.img_size)
        self.new_size = tuple(np.round(self.img_size * self.ratio).astype(np.int32))

    def reshape(self, bottom, top):
        # tmp_shape = list(bottom[0].data.shape)
        # tmp_shape[2] = self.new_size[0]  # new height
        # tmp_shape[3] = self.new_size[1]  # new width
        # # No registered converter was able to produce a C++ rvalue of type int
        # # from this Python object of type numpy.int32
        # top[0].reshape(*tmp_shape)
        top[0].reshape(*[bottom[0].data.shape[0], bottom[0].data.shape[1],
                       int(self.new_size[0]), int(self.new_size[1])])

    def forward(self, bottom, top):
        for k in range(bottom[0].data.shape[0]):
            img_bottom = bottom[0].data[k, ...]
            img = img_bottom.transpose(1, 2, 0)  # h x w x 3
            img_resized = cv2.resize(img, (self.new_size[1], self.new_size[0]))
            top[0].data[k, ...] = img_resized.transpose(2, 0, 1) - self.mean  # 3 x h x w

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            raise NotImplementedError(
                "Backward pass not supported with this implementation")
        else:
            pass
