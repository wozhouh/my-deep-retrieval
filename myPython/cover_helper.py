# -*- coding: utf-8 -*-

# Python class with common operations on the cover dataset

# import download_helper  # used in Python 3, annotated first
import argparse
import os
import random
import cv2
import shutil
from region_generator import *


class CoverDataset:
    def __init__(self, root_dir):
        self.cls_file = os.path.join(root_dir, 'demo_shortv.txt')
        self.root_dir = root_dir
        self.training_dir = os.path.join(root_dir, 'training')
        self.test_dir = os.path.join(root_dir, 'test')
        self.clean_dir = os.path.join(root_dir, 'clean')
        self.cls_dir = os.path.join(self.clean_dir, 'cls')
        self.csmid_test_list = []  # list in list
        self.csmid_training_list = []  # list in list
        self.csmid_list = []
        self.cls_num = 0
        self.S = None
        self.L = 2
        # calculates the mean on training set in advance
        self.mean = np.array([117.80904, 130.27611, 134.65074], dtype=np.float32)[:, None, None]
        self.dataset = []  # list of image names
        self.q_fname = []  # list of image names
        self.a_fname = []  # list of image names in list of classes
        self.a_idx = []  # list of image indices in list of classes
        self.num_queries = 0
        self.num_dataset = 0

        if not os.path.exists(self.training_dir):
            os.makedirs(self.training_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        if not os.path.exists(self.clean_dir):
            os.makedirs(self.clean_dir)
        if not os.path.exists(self.cls_dir):
            os.makedirs(self.cls_dir)

    '''
    [     0      0 195170  59793  27347  15007   9328   6156   4267   3146
       5664   2624   1336    875    581    429    272    246    193    158
        125     98     75     70     68     51     49     31     35     33
         23     23     15     26     15      7     10     12     12      9
          9      4     10      3      6      3      7      5      4     10
          1      5      2      1      1      4      1      4      2      0
          2      3      2      1      4      0      1      0      2      1
          0      0      1      1      0      1      0      1      1      0
          0      0      0      0      1      0      0      1      1      0
          0      0      0      0      0      0      0      1      0      0]
    '''
    # return an array with index indicating frequency and value telling the number of items as above
    def count_items(self):
        cnt_max = 100  # Actually the max count of the cover dataset is 97 exactly
        cnt = np.zeros(cnt_max, dtype=int)
        f = open(self.cls_file, 'r')
        for line in f:
            item_num = len(line.split(' '))
            cnt[item_num] += 1
        return cnt

    # put csmid of a certain class into test set list whose item number is between cnt_min and cnt_max
    def get_csmid_list(self, cnt_min=20, cnt_max=24):
        f = open(self.cls_file, 'r')
        for k, line in enumerate(f.readlines()):
            temp_list = line.split(' ')
            item_list = temp_list[1: -1]  # add the csmid on the line except the first and the last one
            item_list.append((temp_list[0].split('\t'))[-1])  # add the first csmid (exclude the prefix id)
            item_list.append((temp_list[-1].split('\n'))[0])  # add the first csmid (exclude the '\n')
            item_num = len(item_list)
            if item_num != len(temp_list):
                print("WARNING: csmid missed at line %d" % k)
            if (item_num >= cnt_min) & (item_num <= cnt_max):
                self.csmid_test_list.append(item_list)
                self.cls_num += 1  # 436 in total
            else:
                self.csmid_training_list.append(item_list)

    # Download the images for the test set
    # 'start' used for recover downloading when stop at class 'start'
    def make_test_set(self, start=0):
        assert self.cls_num > 0, 'list of images for test set still empty'
        csmid_test_waiting = self.csmid_test_list[start:]
        for k, imgs in enumerate(csmid_test_waiting):
            k += start
            print('Downloading images of class %s' % str(k))
            cls = os.path.join(self.cls_dir, str(k))
            if not os.path.exists(cls):
                os.makedirs(cls)
            else:
                for img in os.listdir(cls):
                    os.remove(os.path.join(cls, img))
            for idx, img_csmid in enumerate(imgs):
                img_name = 'cls' + str(k) + '_img' + str(idx) + '.jpg'
                cls_path = os.path.join(cls, img_name)
                # download_helper.get_img_by_cmsid(img_csmid, cls_path)

    # remove the image files not from downloading originally
    def clean_test_set(self):
        for k in range(self.cls_num):
            cls = os.path.join(self.cls_dir, str(k))
            for img in os.listdir(cls):
                if not img.startswith('cls'):
                    os.remove(os.path.join(cls, img))

    # After mannually clean the dataset, build the clean test set for validation
    def make_clean_set(self, temp_dir):
        for i, c in enumerate(os.listdir(temp_dir)):
            src_dir = os.path.join(temp_dir, c)
            dst_dir = os.path.join(self.cls_dir, str(i))
            img_dir = os.path.join(self.clean_dir, 'img')
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            for k, img in enumerate(os.listdir(src_dir)):
                img_src_path = os.path.join(src_dir, img)
                img_dst_path = os.path.join(dst_dir, 'cls'+str(i)+'_img'+str(k)+'.jpg')
                img_dir_path = os.path.join(img_dir, 'cls'+str(i)+'_img'+str(k)+'.jpg')
                open(img_dst_path, 'wb').write(open(img_src_path, 'rb').read())
                open(img_dir_path, 'wb').write(open(img_src_path, 'rb').read())

    # Download the images for the training set
    # 'start' used for recover downloading when stop at image 'start'
    # ${CAFFE_ROOT}/build/tools/convert_imageset ${IMAGE_ROOT}/ ${DATA_ROOT}/${IMAGE_LIST} ${DATA_ROOT}/${LMDB_NAME}
    def make_training_set(self, start=0):
        assert len(self.csmid_training_list) > 0, 'list of images for training set still empty'
        csmid_list = []
        for cls in self.csmid_training_list:
            csmid_list.extend(cls)
        csmid_training_waiting = csmid_list[start:]
        img_dir = os.path.join(self.training_dir, 'img')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        for k, img_id in enumerate(csmid_training_waiting):
            k += start
            print('Downloading image %d' % k)
            img_name = 'img' + str(k) + '.jpg'
            img_path = os.path.join(img_dir, img_name)
            # delete if exist
            if os.path.isfile(img_path):
                os.remove(img_path)
            # download_helper.get_img_by_cmsid(img_id, img_path)

    # Generates two file recording files for query (1 in each class) and answer (the others) respectively
    def make_queries_for_test(self):
        query_file = os.path.join(self.clean_dir, 'query.txt')
        answer_file = os.path.join(self.clean_dir, 'answer.txt')
        f_q = open(query_file, 'w')
        f_a = open(answer_file, 'w')
        queries_dir = os.path.join(self.clean_dir, 'queries')
        dataset_dir = os.path.join(self.clean_dir, 'dataset')
        if not os.path.exists(queries_dir):
            os.makedirs(queries_dir)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # split the images in every class into queries and corresponding answers
        for c in os.listdir(self.cls_dir):
            cls_path = os.path.join(self.cls_dir, c)
            img_list = os.listdir(cls_path)
            img_answer = ''
            if len(img_list) >= 3:
                img_query = (random.sample(img_list, 1))[0]
                open(os.path.join(queries_dir, img_query), 'wb')\
                    .write(open(os.path.join(cls_path, img_query), 'rb').read())
                f_q.write(img_query + '\n')
                for img in img_list:
                    if img != img_query:
                        img_answer += (img+'\t')
                        open(os.path.join(dataset_dir, img), 'wb').write(open(os.path.join(cls_path, img), 'rb').read())
                f_a.write(img_answer + '\n')
            else:
                for img in img_list:
                    open(os.path.join(dataset_dir, img), 'wb').write(open(os.path.join(cls_path, img), 'rb').read())

        f_q.close()
        f_a.close()

    # Initializes the list of queries and answers for future use of precision and mAP calculation
    def get_queries_answer_list(self):
        self.dataset = os.listdir(os.path.join(self.clean_dir, 'dataset'))
        q_lines = open(os.path.join(self.clean_dir, 'query.txt')).readlines()
        a_lines = open(os.path.join(self.clean_dir, 'answer.txt')).readlines()
        for line in q_lines:
            self.q_fname.append(line.strip())
        for line in a_lines:
            a_fname_list = line.strip().split('\t')
            self.a_fname.append(a_fname_list)
            self.a_idx.append([self.dataset.index(i) for i in a_fname_list])

        self.num_queries = len(self.q_fname)
        self.num_dataset = len(self.dataset)

    # Calculates each channel's mean value of RGB images in the training set
    def cal_mean_training_set(self):
        img_sum = np.zeros(3, dtype=np.float32)
        img_dir = os.path.join(self.training_dir, 'img')
        fname = os.listdir(img_dir)
        for f in fname:
            img_path = os.path.join(img_dir, f)
            img = cv2.imread(img_path)
            img_sum += img.mean(axis=0).mean(axis=0)
        return img_sum / len(fname)

    # Classifies the images in the training set into sub-directories
    def cls_training_set(self, num_img_downloaded):
        training_cls_dir = os.path.join(self.training_dir, 'cls')
        if not os.path.exists(training_cls_dir):
            os.makedirs(training_cls_dir)
        img_cls_start = 0
        img_cls_end = 0
        for k, cls in enumerate(self.csmid_training_list):
            img_cls_end += len(self.csmid_training_list[k])
            if img_cls_end > num_img_downloaded:
                break
            cls_path = os.path.join(training_cls_dir, str(k))
            if not os.path.exists(cls_path):
                os.makedirs(cls_path)
            else:
                for f in os.listdir(cls_path):
                    os.remove(os.path.join(cls_path, f))
            for i, idx in enumerate(range(img_cls_start, img_cls_end)):
                img_src_name = 'img' + str(idx) + '.jpg'
                img_dst_name = 'cls' + str(k) + '_img' + str(i) + '.jpg'
                img_src_path = os.path.join(self.training_dir, 'img', img_src_name)
                img_dst_path = os.path.join(cls_path, img_dst_name)
                if os.path.isfile(img_src_path):
                    open(img_dst_path, 'wb').write(open(img_src_path, 'rb').read())
                else:
                    print('image not found: %s' % img_src_name)
            img_cls_start = img_cls_end

    def load_image(self, fname):
        img = cv2.imread(fname)
        if self.S is not None:
            img_size_hw = np.array(img.shape[0:2])
            ratio = float(self.S) / np.max(img_size_hw)
            new_size = tuple(np.round(img_size_hw * ratio).astype(np.int32))
            img = cv2.resize(img, (new_size[1], new_size[0]))
        return img.transpose(2, 0, 1) - self.mean

    def prepare_image_and_grid_regions_for_network(self, img_dir, fname):
        img = self.load_image(os.path.join(self.clean_dir, img_dir, fname))
        all_regions = [get_rmac_region_coordinates(img.shape[1], img.shape[2], self.L)]
        regions = pack_regions_for_network(all_regions)
        return np.expand_dims(img, axis=0), regions

    # Calculates the mean precision when number of prediction is equal to GT
    def cal_precision(self, sim, output_img=True):
        assert len(sim.shape) == 2, 'This is a 2-dim similarity matrix'
        assert sim.shape[0] == self.num_queries, 'number of rows should be equal to number of queries'
        assert sim.shape[1] == self.num_dataset, 'number of columns should be equal to number of dataset'
        q_precision = np.zeros(self.num_queries, dtype=np.float32)
        idx = np.argsort(sim, axis=1)[:, ::-1]
        for q in range(self.num_queries):
            top_k = len(self.a_idx[q])  # choose the top-k prediction
            top_idx = list(idx[q, : top_k])
            cnt_correct = len([i for i in top_idx if i in self.a_idx[q]])  # intersection
            q_precision[q] = float(cnt_correct) / float(top_k)
            # output image to 'cls' directory to make a comparison
            if output_img:
                test_cls_dir = os.path.join(self.clean_dir, 'test-cls')
                if not os.path.exists(test_cls_dir):
                    os.makedirs(test_cls_dir)
                else:
                    shutil.rmtree(test_cls_dir)
                    os.makedirs(test_cls_dir)
                pred_img = [self.dataset[i] for i in top_idx]
                gt_img = [self.dataset[i] for i in self.a_idx[q]]
                test_cls_path = os.path.join(test_cls_dir, str(q))
                for k, im in enumerate(gt_img):
                    src_gt_img = os.path.join(self.clean_dir, 'dataset', im)
                    if not os.path.exists(test_cls_path):
                        os.makedirs(test_cls_path)
                    dst_gt_img = os.path.join(test_cls_path, im)
                    open(dst_gt_img, 'wb').write(open(src_gt_img, 'rb').read())
                for k, im in enumerate(pred_img):
                    src_pred_img = os.path.join(self.clean_dir, 'dataset', im)
                    dst_pred_img = os.path.join(test_cls_path, 'error_' + im)
                    if not os.path.exists(os.path.join(test_cls_path, im)):
                        open(dst_pred_img, 'wb').write(open(src_pred_img, 'rb').read())

        return q_precision.mean(axis=0) * 100.0

    # Calculates the value of mAP according to the standard of VOC2010 and later
    def cal_mAP(self, sim):
        assert len(sim.shape) == 2, 'This is a 2-dim similarity matrix'
        assert sim.shape[0] == self.num_queries, 'number of rows should be equal to number of queries'
        assert sim.shape[1] == self.num_dataset, 'number of columns should be equal to number of dataset'
        q_AP = np.zeros(self.num_queries, dtype=np.float32)
        idx = np.argsort(sim, axis=1)[:, ::-1]
        for q in range(self.num_queries):
            cnt_correct_last = 0
            q_ans = self.a_idx[q]
            recall = np.zeros(len(q_ans), dtype=np.float32)
            for d in range(self.num_dataset):
                top_k = d+1
                top_idx = list(idx[q, : top_k])
                cnt_correct = len([i for i in top_idx if i in q_ans])
                assert cnt_correct >= cnt_correct_last
                assert cnt_correct <= len(q_ans)
                if cnt_correct > cnt_correct_last:
                    recall[cnt_correct-1] = float(cnt_correct) / float(top_k)  # precision under the given recall
                    cnt_correct_last = cnt_correct
                if cnt_correct == len(q_ans):
                    break
            # calculates the maximum precision when no less than given recall
            recall_max = np.zeros(len(q_ans), dtype=np.float32)
            for r in range(len(q_ans)):
                recall_max[r] = np.max(recall[r:])
            q_AP[q] = recall_max.mean(axis=0)
            print("AP of query %d: %f" % (q, q_AP[q]))  # print for checking as the function is too slow ...
        return q_AP.mean(axis=0) * 100.0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='build the cover dataset')
    parser.add_argument('--root_dir', type=str, required=False, help='file path to the txt')
    parser.set_defaults(root_dir='/home/processyuan/data/cover/')
    args = parser.parse_args()

    cover_dataset = CoverDataset(args.root_dir)

    # # class with item number between (20, 24) is put into test set while the others into training set
    # cover_dataset.get_csmid_list(cnt_min=20, cnt_max=24)

    # # Download the images for the test set
    # cover_dataset.make_test_set(start=0)

    # # Download the images for the training set
    # # img1652.jpg: URLerror
    # cover_dataset.make_training_set(start=39510)

    # # Classify the training set
    # cover_dataset.cls_training_set(num_img_downloaded=64200)

    # # Build the clean validation set
    # cover_dataset.make_clean_set(os.path.join(args.root_dir, 'clean', 'temp'))
    # cover_dataset.make_queries_for_test()
    # cover_dataset.get_queries_answer_list()
