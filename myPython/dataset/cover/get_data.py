# -*- coding: utf-8 -*-

# import data_helper
import numpy as np
import argparse
import os
import random
import cv2



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


# return an array with index indicating frequency and value telling the corresponding count as above
def count_items(f):
    cnt_max = 100  # Actually the max count of the cover dataset is 97 exactly
    cnt = np.zeros(cnt_max, dtype=int)
    for line in f:
        item_num = len(line.split(' '))
        cnt[item_num] += 1
    return cnt


# given a row in the annotation file, return a list of csmid in a class
def get_item_from_row(row):
    items = row[1: -1]  # add the csmid except the first and the last one
    items.append((row[0].split('\t'))[-1])  # add the first csmid (exclude the prefix id)
    items.append((row[-1].split('\n'))[0])  # add the first csmid (exclude the '\n')
    return items


# put csmid of a certain class into test set list whose item number is between cnt_min and cnt_max
def get_item_list(f, cnt_min=20, cnt_max=24):
    csmid_test_list = []
    csmid_training_list = []
    for k, line in enumerate(f):
        temp_list = line.split(' ')
        item_list = get_item_from_row(temp_list)  # clean the row and get the csmid list
        item_num = len(item_list)
        if item_num != len(temp_list):
            print("WARNING: csmid missed at line %d" % k)
        if (item_num >= cnt_min) & (item_num <= cnt_max):
            csmid_test_list.append(item_list)
        else:
            csmid_training_list.extend(item_list)

    return csmid_training_list, csmid_test_list


# Download the images for the test set
# 'start' used for recover downloading when stop at class 'start'
def make_test_set(csmid, test_dir, cls_dir, start=0):
    if start != 0:
        csmid = csmid[start:]
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(cls_dir):
        os.makedirs(cls_dir)
    for k, imgs in enumerate(csmid):
        k += start
        print('Downloading images of class %s' % str(k))
        cls = os.path.join(cls_dir, str(k))
        if not os.path.exists(cls):
            os.makedirs(cls)
        else:
            for ff in os.listdir(cls):
                os.remove(os.path.join(cls, ff))
        # for i, img_id in enumerate(imgs):
        #     img_name = 'cls' + str(k) + '_img' + str(i) + '.jpg'
        #     cls_path = os.path.join(cls, img_name)
        #     data_helper.get_img_by_cmsid(img_id, cls_path)

    make_queries_for_test(cls_dir, test_dir, len(csmid), num_query=2)


# Download the images for the training set
# 'start' used for recover downloading when stop at image 'start'
def make_training_set(csmid, training_dir, start=0):
    # csmid = csmid[start:]
    # for k, img_id in enumerate(csmid):
    #     k += start
    #     print('Downloading image %d' % k)
    #     img_name = 'img' + str(k) + '.jpg'
    #     img_path = os.path.join(training_dir, 'img', img_name)
    #     # delete if exist
    #     if os.path.isfile(img_path):
    #         os.remove(img_path)
    #     data_helper.get_img_by_cmsid(img_id, img_path)

    # ${CAFFE_ROOT}/build/tools/convert_imageset ${IMAGE_ROOT}/ ${DATA_ROOT}/${IMAGE_LIST} ${DATA_ROOT}/${LMDB_NAME}
    f_txt = open(os.path.join(training_dir, 'train.txt'), 'w')
    training_img = os.listdir(training_dir)
    random.shuffle(training_img)
    for f in training_img:
        f_txt.write(f + ' 0\n')
    f_txt.close()


# Generates two file recording files for query (2 in each class) and answer (the others) respectively
def make_queries_for_test(cls_dir, test_dir, cls_num, num_query=2):
    query_file = os.path.join(test_dir, 'query.txt')
    answer_file = os.path.join(test_dir, 'answer.txt')
    f_q = open(query_file, 'w')
    f_a = open(answer_file, 'w')

    queries_dir = os.path.join(test_dir, 'queries')
    dataset_dir = os.path.join(test_dir, 'dataset')
    if not os.path.exists(queries_dir):
        os.makedirs(queries_dir)
    else:
        for f in os.listdir(queries_dir):
            os.remove(os.path.join(queries_dir, f))
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    else:
        for f in os.listdir(dataset_dir):
            os.remove(os.path.join(dataset_dir, f))

    for k in range(cls_num):
        cls = os.path.join(cls_dir, str(k))
        imgs = []
        for f in os.listdir(cls):
            imgs.append(f)
        query = random.sample(imgs, num_query)
        for q in query:
            f_q.write(q + '\n')
            imgs.remove(q)
            open(os.path.join(queries_dir, q), 'wb').write(open(os.path.join(cls, q), 'rb').read())
        for _ in range(num_query):
            for img in imgs:
                f_a.write(img + ' ')
            f_a.write('\n')
        for img in imgs:
            open(os.path.join(dataset_dir, img), 'wb').write(open(os.path.join(cls, img), 'rb').read())

    f_q.close()
    f_a.close()


# Calculates each channel's mean value of RGB images in the training set
def cal_mean_training_set(train_dir):
    img_sum = np.zeros(3, dtype=np.float32)
    fname = os.listdir(train_dir)
    for f in fname:
        img = cv2.imread(os.path.join(train_dir, f))
        img_sum += img.mean(axis=0).mean(axis=0)
    return img_sum / len(fname)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='build the dataset')
    parser.add_argument('--file', type=str, required=False, help='file path to the txt')
    parser.add_argument('--cls_dir', type=str, required=False, help='directory path to classify images for test set')
    parser.add_argument('--test_dir', type=str, required=False,
                        help='directory path to download the images for test set')
    parser.add_argument('--training_dir', type=str, required=False,
                        help='directory path to download the images for training set')
    parser.set_defaults(file='/home/processyuan/NetworkOptimization/cover/demo_shortv.txt')
    parser.set_defaults(cls_dir='/home/processyuan/NetworkOptimization/cover/cls')
    parser.set_defaults(test_dir='/home/processyuan/NetworkOptimization/cover/test')
    parser.set_defaults(training_dir='/home/processyuan/NetworkOptimization/cover/training')
    args = parser.parse_args()

    # class with item number between (20, 24) is put into test set while the others into training set
    csmid_training, csmid_test = get_item_list(open(args.file, 'r'), cnt_min=20, cnt_max=24)

    # # print for checking
    # print(csmid_training)  # just a list
    # print(csmid_test)  # list in list
    # print(len(csmid_training))  # number of items for training set
    # print(len(csmid_test))  # number of class for test set

    # # Download the images for the test set
    # make_test_set(csmid_test, args.test_dir, args.cls_dir, start=0)

    # # Download the training set
    # # img1652: URLerror
    make_training_set(csmid_training, args.training_dir, start=39510)

