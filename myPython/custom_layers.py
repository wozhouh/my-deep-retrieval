# -*- coding: utf-8 -*-

# self-defined Python layers

import os
import caffe
import yaml
import random
import cv2
import numpy as np
import Queue
import region_generator


# Layer that performs normalization to the input features blob
class NormalizeLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'
        self.eps = 1e-8  # eps added to ganrantee the numerical stability

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


# Layer that sums up the bottom blob along axis=0
class AggregateLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'
        params = yaml.load(self.param_str_)
        self.num_rois = params['num_rois']
        self.batch_size = (bottom[0].data.shape[0]) / self.num_rois

    def reshape(self, bottom, top):
        tmp_shape = list(bottom[0].data.shape)
        tmp_shape[0] = self.batch_size
        top[0].reshape(*tmp_shape)

    def forward(self, bottom, top):
        # top[0].data[:] = bottom[0].data.sum(axis=0)
        for k in range(self.batch_size):
            bottom_data = bottom[0].data[k * self.num_rois: (k + 1) * self.num_rois, ...]
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


# Layer that fetches pre-calculated features from .npy for loss calculation when distilling
class FeatureLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'
        params = yaml.load(self.param_str_)
        self.features_npy = params['features']
        self.batch_size = bottom[0].shape[0]
        self.features = np.load(self.features_npy)
        self.dim = self.features.shape[1]
        self.batch_features = np.zeros((self.batch_size, self.dim, 1, 1), dtype=np.float32)

    def reshape(self, bottom, top):
        top[0].reshape(*[self.batch_size, self.dim, 1, 1])

    def forward(self, bottom, top):
        feature_idx = bottom[0].data.reshape(self.batch_size)
        # iterate over a batch
        for k in range(self.batch_size):
            self.batch_features[k, :] = (self.features[int(feature_idx[k]), :]).reshape(self.dim, 1, 1)
        top[0].data[...] = self.batch_features

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            raise NotImplementedError(
                "Backward pass not supported with this implementation")
        else:
            pass


# Layer that generates rigid grid of bottom blob (batch size and number of rois should be given as param_str)
# within the batch, the image size should be the same (which results in the same number of rois)
class RigidGridLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'
        # assert bottom[0].data.shape[0] == 1, 'Batch size is fixed to 1 as the size of images might be different'
        assert bottom[0].data.shape[1] == 3, 'The input should be a 3-channel RGB image in batch x 3 x H x W format'
        params = yaml.load(self.param_str_)
        self.dataset = params['dataset']
        self.num_region = 8  # for regular images, the typical number of rigid regions is 8 so fix it here
        self.dim_rois = 5  # (index, xmin, ymin, xmax, ymax)
        self.img_h = bottom[0].data.shape[2]  # for the cover images, h = 280
        self.img_w = bottom[0].data.shape[3]  # for the cover images, w = 496
        self.batch_size = bottom[0].data.shape[0]  # bottom: (batch_size, channels(3), h(280), w(496))
        self.cover_rois = np.array([[0.,  0.,  0., 279., 279.],
                                    [0., 216., 0., 495., 279.],
                                    [0.,  0.,  0., 185., 185.],
                                    [0., 155., 0., 340., 185.],
                                    [0., 310., 0., 495., 185.],
                                    [0.,  0., 94., 185., 279.],
                                    [0., 155.,94., 340., 279.],
                                    [0., 310.,94., 495., 279.]])
        self.paris_rois = np.array([[0.,  0.,  0., 383., 383.],
                                    [0., 128., 0., 511., 383.],
                                    [0.,  0.,  0., 255., 255.],
                                    [0., 128., 0., 383., 255.],
                                    [0., 256., 0., 511., 255.],
                                    [0.,  0., 128.,255., 383.],
                                    [0., 128.,128.,383., 383.],
                                    [0., 256.,128.,511., 383.]])
        self.landmark_rois = np.array([[0., 0., 0., 287., 287.],
                                    [0., 96., 0., 383., 287.],
                                    [0., 0., 0., 191., 191.],
                                    [0., 96., 0., 287., 191.],
                                    [0., 192., 0., 383., 191.],
                                    [0., 0., 96., 191., 287.],
                                    [0., 96., 96., 287., 287.],
                                    [0., 192., 96., 383., 287.]])

    def reshape(self, bottom, top):
        top[0].reshape(*[self.batch_size * self.num_region, self.dim_rois])

    def forward(self, bottom, top):
        if self.dataset == 'cover':
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
            R = self.cover_rois
        elif self.dataset == 'paris':
            '''
            (1, 3, 384, 512)
            [[0.   0.   0. 383. 383.]
             [0. 128.   0. 511. 383.]
             [0.   0.   0. 255. 255.]
             [0. 128.   0. 383. 255.]
             [0. 256.   0. 511. 255.]
             [0.  0.  128. 255. 383.]
             [0. 128. 128. 383. 383.]
             [0. 256. 128. 511. 383.]]
            '''
            R = self.paris_rois
        elif self.dataset == 'landmark':
            '''
           (1, 3, 288, 384)
           [[0., 0., 0., 287., 287.],
            [0., 96., 0., 383., 287.],
            [0., 0., 0., 191., 191.],
            [0., 96., 0., 287., 191.],
            [0., 192., 0., 383., 191.],
            [0., 0., 96., 191., 287.],
            [0., 96., 96., 287., 287.],
            [0., 192., 96., 383., 287.]]
           '''
            R = self.landmark_rois
        else:
            all_regions = [region_generator.get_rmac_region_coordinates(self.img_h, self.img_w, 2)]
            R = region_generator.pack_regions_for_network(all_regions)  # for the cover images, R,shape = [8, 5]

        # iterate over a batch
        if self.batch_size == 1:
            top[0].data[:] = np.array(R[: self.num_region, :])
        else:
            reg = np.zeros((self.num_region * self.batch_size, self.dim_rois), dtype=np.float32)
            for k in range(self.batch_size):
                R_temp = np.array(R)
                R_temp[:, 0] = k  # image index in the batch
                reg[k * self.num_region: (k + 1) * self.num_region, :] = R_temp
            top[0].data[:] = np.array(reg)

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            raise NotImplementedError(
                "Backward pass not supported with this implementation")
        else:
            pass


# Layer that resizes the image to the given height and width and then substracts the mean value of channels
class ResizeLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'This layer can only have one bottom'
        params = yaml.load(self.param_str_)
        self.h = params['h']
        self.w = params['w']
        self.mean = np.array(params['mean'], dtype=np.float32)[:, None, None]

    def reshape(self, bottom, top):
        top[0].reshape(*[bottom[0].data.shape[0], bottom[0].data.shape[1], self.h, self.w])

    def forward(self, bottom, top):
        # iterate over a batch
        for k in range(bottom[0].data.shape[0]):
            img_bottom = bottom[0].data[k, ...]
            img = img_bottom.transpose(1, 2, 0)  # h x w x 3
            img_resized = cv2.resize(img, (self.w, self.h))
            top[0].data[k, ...] = img_resized.transpose(2, 0, 1) - self.mean  # 3 x h x w

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            raise NotImplementedError(
                "Backward pass not supported with this implementation")
        else:
            pass


# A data layer that fetches the images and feeds the triplet siamese network
# Fully shuffling the data but not the hard negative mining
# !!! Deprecated Layer !!! --> turn to BinDataLayer instead
class TripletDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 0, 'Data layer should not have a bottom for input'
        assert len(top) == 3, 'Data layer for the triplet-siamese network should have 3 tops'
        params = yaml.load(self.param_str_)
        self.batch_size = params['batch_size']
        self.cls_dir = params['cls_dir']
        # self.useless_dir = os.path.join(self.cls_dir, 'useless')
        self.mean = np.array(params['mean'], dtype=np.float32)[:, None, None]
        self.cls = os.listdir(self.cls_dir)   # list of classes
        if 'junk' in self.cls:
            self.cls.remove('junk')  # except 'junk'
        self.cls_ind = len(self.cls) - 1  # init: suppose an epoch is done
        self.img = []  # list of images within the current class
        self.img_ind = 0  # index of the images in process

    # Fix the image shape here to the Paris dataset (288, 384, 3)
    def reshape(self, bottom, top):
        top[0].reshape(*[self.batch_size, 3, 288, 384])
        top[1].reshape(*[self.batch_size, 3, 288, 384])
        top[2].reshape(*[self.batch_size, 3, 288, 384])

    def forward(self, bottom, top):
        if self.img_ind >= len(self.img):
            self.cls_ind += 1
            if self.cls_ind == len(self.cls):
                random.shuffle(self.cls)
                self.cls_ind = 0
                # print("INFO: an epoch done.")
            self.img = os.listdir(os.path.join(self.cls_dir, self.cls[self.cls_ind]))
            random.shuffle(self.img)
            self.img_ind = 0

        # fetch the images in process
        t_diff = self.batch_size + self.img_ind - len(self.img)
        if t_diff <= 0:
            t_img_name = self.img[self.img_ind: self.img_ind + self.batch_size]
        else:
            t_img_name = self.img[self.img_ind: len(self.img)]
            t_img_name.extend(self.img[: t_diff])  # pad the rest with the images from the beginning

        # randomly sample a positive image from the same class
        p_img_dir = os.path.join(self.cls_dir, self.cls[self.cls_ind])
        p_diff = self.batch_size
        p_img_name = []  # a positive image which is in the same class as t_img
        while p_diff > 0:
            p_img_name_temp = random.sample(self.img, p_diff)
            for i in p_img_name_temp:
                if i in t_img_name or i in p_img_name:
                    p_img_name_temp.remove(i)
            p_img_name.extend(p_img_name_temp)
            p_diff = self.batch_size - len(p_img_name)

        # randomly sample a negative image from a different class
        n_cls = -1
        while n_cls == self.cls_ind or n_cls == -1:
            n_cls = random.randint(0, len(self.cls)-1)
        n_img_dir = os.path.join(self.cls_dir, self.cls[n_cls])
        n_img_name = random.sample(os.listdir(n_img_dir), self.batch_size)

        # load the images
        t_img_temp = [cv2.imread(os.path.join(p_img_dir, t_img_name[k]))
                      for k in range(self.batch_size)]
        t_img = [(t_img_temp[k].transpose(2, 0, 1) - self.mean) for k in range(self.batch_size)]
        p_img_temp = [cv2.imread(os.path.join(p_img_dir, p_img_name[k]))
                      for k in range(self.batch_size)]
        p_img = [(p_img_temp[k].transpose(2, 0, 1) - self.mean) for k in range(self.batch_size)]
        n_img_temp = [cv2.imread(os.path.join(n_img_dir, n_img_name[k]))
                      for k in range(self.batch_size)]
        n_img = [(n_img_temp[k].transpose(2, 0, 1) - self.mean) for k in range(self.batch_size)]

        # iterate over a batch
        for k in range(self.batch_size):
            top[0].data[k, ...] = t_img[k]
            top[1].data[k, ...] = p_img[k]
            top[2].data[k, ...] = n_img[k]
        self.img_ind += self.batch_size

    # No need for a data layer to implement the 'backward' function
    def backward(self, top, propagate_down, bottom):
        pass


# A data layer that fetches images from the same and different class to form a batch (half by half)
class BinDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 0, 'Data layer should not have a bottom for input'
        assert len(top) == 2, 'BinDataLayer should have 2 tops'
        params = yaml.load(self.param_str_)
        self.batch_size = params['batch_size']
        self.cls_dir = params['cls_dir']
        self.dataset = params['dataset']
        self.mean = np.array(params['mean'], dtype=np.float32)[:, None, None]
        self.cls = os.listdir(self.cls_dir)  # list of classes
        self.img_queue = Queue.Queue(maxsize=0)  # queue for fetching iamges in an epoch
        self.label_queue = Queue.Queue(maxsize=0)  # queue for corresponding labels in an epoch
        self.ind = 0

    # Fix the image shape here to the Paris dataset (288, 384, 3)
    def reshape(self, bottom, top):
        if self.dataset == 'landmark':
            top[0].reshape(*[self.batch_size, 3, 288, 384])
        elif self.dataset == 'paris':
            top[0].reshape(*[self.batch_size, 3, 384, 512])
        else:
            top[0].reshape(*[self.batch_size, 3, 280, 496])
        top[1].reshape(*[self.batch_size, 1, 1, 1])  # labels

    def forward(self, bottom, top):
        if self.label_queue.empty():
            print('INFO: An epoch is done.')
            self.get_epoch_data()
        img_path_list = self.img_queue.get()
        labels_list = self.label_queue.get()
        img_temp = np.array([cv2.imread(i) for i in img_path_list])
        labels = np.array(labels_list).reshape(self.batch_size, 1, 1, 1)
        img = [(i.transpose(2, 0, 1) - self.mean) for i in img_temp]
        top[0].data[...] = img
        top[1].data[...] = labels

    # No need for a data layer to implement the 'backward' function
    def backward(self, top, propagate_down, bottom):
        pass

    # when an epoch is done, shuffle the data
    def get_epoch_data(self):
        img_queue_temp = []  # list of list for fetching images in an epoch
        labels_queue_temp = []  # list of list for corresponding labels in an epoch
        pos_num = self.batch_size / 4  # number of positive samples in the batch
        neg_num = self.batch_size - pos_num  # number of negative samples in the batch
        for c in self.cls:
            img_ind = 0  # index of the images in process
            cls_path = os.path.join(self.cls_dir, c)
            cls_except = list(self.cls)
            cls_except.remove(c)
            img = os.listdir(cls_path)
            random.shuffle(img)
            while pos_num + img_ind <= len(img):
                # fetch the images within the same class
                img_path_list = []
                labels_list = []
                for k in range(pos_num):
                    img_path = os.path.join(cls_path, img[img_ind + k])
                    img_path_list.append(img_path)
                    labels_list.append(int(c))

                # randomly sample a negative image from different classes
                neg_img_cls = random.sample(cls_except, neg_num)
                for n_cls in neg_img_cls:
                    n_img_dir = os.path.join(self.cls_dir, n_cls)
                    n_img_name = (random.sample(os.listdir(n_img_dir), 1))[0]
                    img_path_list.append(os.path.join(n_img_dir, n_img_name))
                    labels_list.append(int(n_cls))
                img_ind += pos_num

                # load the images
                img_queue_temp.append(img_path_list)
                labels_queue_temp.append(labels_list)

        # shuffle the images and the corresponding labels in he same order
        randnum = random.randint(0, 1000000)
        random.seed(randnum)
        random.shuffle(img_queue_temp)
        random.seed(randnum)
        random.shuffle(labels_queue_temp)

        # push into the queue
        for i, k in enumerate(labels_queue_temp):
            self.img_queue.put(img_queue_temp[i])
            self.label_queue.put(k)
