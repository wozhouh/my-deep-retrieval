# my Deep Retrieval

This respository origins from a [public implementation](https://github.com/figitaki/deep-retrieval) of [Deep Image Retrieval](https://link.springer.com/content/pdf/10.1007%2F978-3-319-46466-4_15.pdf). Although learned form a dataset of landmarks, the global representation it extracts from images is a robust solution for retrieving same-origin frames from a large data-pool of short videos. 

To be specific, we measure the similarity of two images by calculating the cosine distance (after normalized) of the two embeddings extracted by the model of CNN and R-MAC. We find that a reasonable similarity threshold is helpful to the selection of the frames that look alike, because there is usually a gap of similarity scores between the same-origin frames and others. The method is especially useful when dealing with videos of something like daily events but not so robust with those of portraits and games, which might call for the fine-grained recognition.

|query|top1|top2|top3|top4|top5|
|---|---|---|---|---|---|
|![](images/20180314V17VZ000_1.000.jpg)|![](images/20180518V1ES6200_0.968.jpg)|![](images/20180421V1F4P900_0.679.jpg)|![](images/20180524V1YLTX00_0.611.jpg)|![](images/20180507V06SDP00_0.605.jpg)|![](images/20180320V0DV1300_0.604.jpg)|
|similarity|0.968|0.679|0.611|0.605|0.604|

|query|top1|top2|top3|top4|top5|
|---|---|---|---|---|---|
|![](images/20180205V0R8GH00_1.000.jpg)|![](images/20180314V1OE6200_0.630.jpg)|![](images/20180517V0M2VO00_0.619.jpg)|![](images/20180522V0H77Y00_0.617.jpg)|![](images/20180529V0T9DG00_0.605.jpg)|![](images/20180608V174Y400_0.596.jpg)|
|similarity|0.630|0.619|0.617|0.605|0.596|

We notice that the [original implementation](https://github.com/wozhouh/my-deep-retrieval/blob/master/test.py) provides an option of multi-resolution (0.5x/1.0x/1.5x), and we find that it does improve the performance on Oxford, which is measured by [mAP](https://github.com/wozhouh/my-deep-retrieval/blob/master/eval/compute_ap.cpp). However, there is two points that is not so desirable. Firstly, the multi-resolution calculatio is too redundant to afford. Secondly, the embeddings have a dimensionality of 2048, making it expensive for storage and further calculation. So we have the following attempts to improve.

### Offline PCA
It is a natural idea whether ultilizing the feature maps of different sizes extracted from the middle layers could improve the descriptor and take similar effects to the multi-resolution (something like SSD). Firstly we concat the features from middle layers directly with the one from the last layer and perform offline PCA (optionally) to get the descriptor, and find that it helps, although only slightly. The results below of offline PCA, which is measured by mAP on Oxford dataset, serves as our baseline and the code can be refered [here](https://github.com/wozhouh/my-deep-retrieval/tree/master/myPython/offline). As the effects of offline PCA are not remarkable enough, and it might increase the dimensionality so as to obtain enhanced performances, which is not desirable. 

|model|resolution|PCA or not|number of feature maps|mAP|
|----|----|----|----|----|
|original|512|Y|1|81.11|
|three-resolution|512|Y|1|82.88|
|original|512|Y|1 (middle)|68.33|
|single-resolution|512|N|1|76.75|
|single-resolution|512|N|2|76.48|
|single-resolution|512|N|4|76.61|
|single-resolution|512|Y|2|81.62|
|single-resolution|512|Y|4|81.78|

|model|resolution|PCA or not|number of feature maps|dim|mAP|
|----|----|----|----|----|----|
|original|512|Y|1|2048|81.11|
|single-resolution|512|Y|2|3072|81.62|
|single-resolution|512|Y|2|2048|80.33|
|single-resolution|512|Y|4|3840|81.78|
|single-resolution|512|Y|4|2048|80.99|

### Knowledge Distiiling
For further enhancemnet, we refer to knowledge distiiling to transfer the ability of 3-resoltion model to single-resolution. For this task, we have [Paris dataset](https://github.com/wozhouh/my-deep-retrieval/blob/master/myPython/paris_helper.py) and [Landmark dataset](https://github.com/wozhouh/my-deep-retrieval/blob/master/myPython/landmark_helper.py) for training and have [Oxford dataset](https://github.com/wozhouh/my-deep-retrieval/blob/master/myPython/oxford_helper.py) and [cover dataset](https://github.com/wozhouh/my-deep-retrieval/blob/master/myPython/cover_helper.py) for validation and test. We use a teacher model of 3-resoltion to [extract the descriptors](https://github.com/wozhouh/my-deep-retrieval/blob/master/myPython/convert_image2features_multires.py) of images on  the training set, and train a student model with loss of distance between the teacher descriptors and the student descriptors. Note that training in the way of knowledge distilling and the following metric learning will use the custom implementation of some auxiliary [layers](https://github.com/wozhouh/my-deep-retrieval/blob/master/myPython/custom_layers.py) written in Python. Considering that the single resolution model has no access to the performance gain of multiple reception fields, so we test different [architectures](https://github.com/wozhouh/my-deep-retrieval/tree/master/proto/distilling) to fuse features from middle layers, and the results are as follow (measured by mAP on Oxford dataset of unified-resolution 512x and cover dataset). Only the original architecture trained by distilling is acceptable, and we guess that the other architectures is unable to maximize the performance due to a small training set (3w+) that is different from the one for training the original model.

|model|mAP(uni-Oxford)|mAP(cover)|weights|
|----|----|----|----|
|original|80.45|98.87|187MB|
|three-resolution|82.90|98.97|187MB|
|original fine-tuned|80.52|99.03|187MB|
|single-resolution: pca-concat-pca|2048|79.45|98.82|217MB|
|single-resolution: 1x1conv-eltwise-pca|78.75|98.80|196MB|
|single-resolution: concat-pca-relu|74.18|/|196MB|

### Metric Learning
As the dimension of desriptors (2048 float) is too large for storage and later calculation, which we wish to cut it down to 512 or smaller. So we train the extractor again, but this time with the loss of metric learning. we use triplet loss firstly but find it hard to converge without hard negative mining and large enough GPU memory. As a result we turn to [Lifted Structured Feature Embedding](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16). You can find the configuration of trained model [here](https://github.com/wozhouh/my-deep-retrieval/tree/master/proto/triplet/pca512). After metric-learning, we stack multi-resoltion and perform knowledge distiiling again to make further improvement. Finally we get a [descriptor-extractor](https://github.com/wozhouh/my-deep-retrieval/tree/master/proto/distilling/pca512) with reduced-dimensionality, although with slight loss of performance compared to the original model. 

We validates the models mentioned above on the cover dataset by percision (number of same-origin ones) and recall (number of unrelevant ones) on 524 groups of images (5k) of cover dataset. Also, you can visualize the clusters of top-k retrieved frames [here](https://github.com/wozhouh/my-deep-retrieval/blob/master/myPython/test/test_SimilarCluster.py).

|triplet512/threshold|0.70|0.71|0.72|0.73|0.74|0.75|0.76|0.77|0.78|0.79|0.80|
|----|----|----|----|----|----|----|----|----|----|----|----|
|same-origin|1547|1530|1509|1484|1464|1432|1408|1383|1338|1294|1254|
|unrelevant|3677|3658|3600|3478|3268|2942|2587|2177|1829|1515|1228|

|distilling/threshold|/|/|0.56|0.57|0.58|0.59|0.60|0.61|0.62|0.63|0.64|
|----|----|----|----|----|----|----|----|----|----|----|----|
|same-origin|-------|-------|1504|1488|1467|1437|1417|1390|1369|1342|1298|
|unrelevant|-------|-------|3127|2773|2436|2098|1802|1521|1316|1131|948|

|multires/threshold|/|/|0.65|0.66|0.67|0.68|0.69|0.70|0.71|0.72|0.73|
|----|----|----|----|----|----|----|----|----|----|----|----|
|same-origin|-------|-------|1514|1493|1460|1429|1396|1367|1335|1313|1272|
|unrelevant|-------|-------|3393|2951|2464|2016|1647|1361|1121|945|781|

Actually we also try another way to reduce the dimensionality, that is, train the model with [pair loss](https://github.com/wozhouh/my-deep-retrieval/tree/master/proto/reduce) by calculating the distance of all the pairs within a batch, and it reaches similar effects as "triplet512" above, which is about a mAP of 78.5x on uni-oxford dataset.

### Dataset && Code
The dataset used here includes [Oxford dataset](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/), [Paris dataset](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/), [Landmark dataset](https://www.kaggle.com/c/landmark-recognition-challenge) and cover dataset. 

The Python code is under the path 'myPython/' as follow:

```
├── myPython/
│   ├── custom_layers.py              /* custom Python layers
│   ├── model_tools.py                /* visualization and modification of Caffe weights
│   ├── region_generator.py           /* RoI generator given the resolution
│   ├── check_on_cover.py             /* test of precision and recall under different similarity threshold on cover dataset
│   ├── train.py                      /* train using Caffe PythonAPI
│   ├── dataset_helper/               /* operations on several dataset
│   │   ├── cover_helper.py           /* cover
│   │   ├── landmark_helper.py        /* Landmark
│   │   ├── paris_helper.py           /* Paris
│   │   ├── oxford_helper.py          /* Oxford
│   │   ├── download_helper.py        /* download images by cmsid
│   ├── test/                         /* evaluation
│   │   ├── test_on_oxford.py         /* Oxford
│   │   ├── test_on_paris.py          /* Paris
│   │   ├── test_on_landmark.py       /* Landmark
│   │   ├── test_on_cover.py          /* cover
│   │   ├── test_SimilarCluster.py    /* visualization of the similar clusters
│   ├── convert/                      /* convert the images into embeddings
│   ├── offline/                      /* offline PCA experiments
```

The configuration of models and their weights are under the path 'proto/' and 'caffemodel/' as follow：
```
├── proto/
│   ├── offline/                      /* offline PCA experiments
│   ├── distilling/                   /* knowledge distilling（including the 3-resolution teacher model）
│   ├── triplet/                      /* training with triplet-loss
│   ├── reduce/                       /* training with pair loss (parallel for the RoI features，serial for the embeddings)
│   ├── deploy_resnet101.prototxt     /* original model
│   ├── deploy_resnet101_normpython.prototxt     /* original model with Python normalize_layer
```

# Deep Retrieval

This package contains the pretrained ResNet101 model and evaluation script for the method proposed in the following papers:

* *Deep Image Retrieval: Learning global representations for image search.* A. Gordo, J. Almazan, J. Revaud, and D. Larlus. In ECCV, 2016
* *End-to-end Learning of Deep Visual Representations for Image Retrieval.* A. Gordo, J. Almazan, J. Revaud, and D. Larlus. CoRR abs/1610.07940, 2016

### Dependencies:
 - Caffe
 - Region of Interest pooling layer (ROIPooling). This is the same layer used by fast RCNN and faster RCNN. A C++ implementation can be found in https://github.com/BVLC/caffe/pull/4163
 - L2-normalization layer (Normalize). Implemented in C++ in https://github.com/happynear/caffe-windows. As an alternative, we provide a python implementation of this layer that produces the same results, but is less efficient and does not implement backpropagation.


### Datasets
The evaluation script is prepared to work on the Oxford 5k and Paris 6k datasets. To set up the datasets:

```sh
mkdir datasets
cd datasets
```

**Evaluation script:**
```sh
mkdir evaluation
cd evaluation
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp
sed -i '6i#include <cstdlib>' compute_ap.cpp # Add cstdlib, as some compilers will produce an error otherwise
g++ -o compute_ap compute_ap.cpp
cd ..
```

**Oxford:**
```sh
mkdir -p Oxford
cd Oxford
mkdir jpg lab
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz
tar -xzf oxbuild_images.tgz -C jpg
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz
tar -xzf gt_files_170407.tgz -C lab
cd ..
```

**Paris**
```sh
mkdir -p Paris
cd Paris
mkdir jpg lab tmp
# Images are in a different folder structure, need to move them around
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz
tar -xzf paris_1.tgz -C tmp
tar -xzf paris_2.tgz -C tmp
find tmp -type f -exec mv {} jpg/ \;
rm -rf tmp
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_120310.tgz
tar -xzf paris_120310.tgz -C lab
cd ..
cd ..
```

## Usage
```
$ python test.py

usage: test.py [-h] --gpu GPU --S S --L L --proto PROTO --weights WEIGHTS
               --dataset DATASET --dataset_name DATASET_NAME --eval_binary
               EVAL_BINARY --temp_dir TEMP_DIR [--multires] [--aqe AQE]
               [--dbe DBE]

G: gpu id
S: size to resize the largest side of the images to. The model is trained with S=800, but different values may work better depending on the task.
L: number of levels of the rigid grid. Model was trained with L=2, but different levels (e.g. L=1 or L=3) may work better on other tasks.
PROTO: path to the prototxt. There are two prototxts included.
  deploy_resnet101.prototxt relies on caffe being compiled with the normalization layer.
  deploy_resnet101_normpython.prototxt does not have that requirement as it relies on the python implementation, but it may be slower as it is done on the cpu and does not implement backpropagation.
WEIGHTS: path to the caffemodel
DATASET: path to the dataset, for Oxford and Paris it is the directory that contains the jpg and lab folders.
DATASET_NAME: either Oxford or Paris
EVAL_BINARY: path to the compute_ap binary provided with Oxford and Paris used to compute the ap scores
TEMP_DIR: a temporary directory to store features and scores
```

Note that this model does not implement the region proposal network.

## Examples
Adjust paths as necessary:

**Rigid grid, no multiresolution, no query expansion or database side feature augmentation:**

```sh
python test.py --gpu 0 --S 800 --L 2 --proto deploy_resnet101_normpython.prototxt --weights model.caffemodel --dataset datasets/Oxford --eval_binary datasets/evaluation/compute_ap --temp_dir tmp --dataset_name Oxford
```
Expected accuracy: `84.09`

```sh
python test.py --gpu 0 --S 800 --L 2 --proto deploy_resnet101_normpython.prototxt --weights model.caffemodel --dataset datasets/Paris --eval_binary datasets/evaluation/compute_ap --temp_dir tmp --dataset_name Paris
```
Expected accuracy: `93.57`

**Rigid grid, multiresolution, no query expansion or database side feature augmentation:**
```sh
python test.py --gpu 0 --S 800 --L 2 --proto deploy_resnet101_normpython.prototxt --weights model.caffemodel --dataset datasets/Oxford --eval_binary datasets/evaluation/compute_ap --temp_dir tmp --dataset_name Oxford --multires
```
Expected accuracy: `86.07`

```sh
python test.py --gpu 0 --S 800 --L 2 --proto deploy_resnet101_normpython.prototxt --weights model.caffemodel --dataset datasets/Paris --eval_binary datasets/evaluation/compute_ap --temp_dir tmp --dataset_name Paris --multires
```
Expected accuracy: `94.53`

**Rigid grid, multiresolution, query expansion (k=1) and database side feature augmentation (k=20):**
```sh
python test.py --gpu 0 --S 800 --L 2 --proto deploy_resnet101_normpython.prototxt --weights model.caffemodel --dataset datasets/Oxford --eval_binary datasets/evaluation/compute_ap --temp_dir tmp --dataset_name Oxford –multires --aqe 1 --dbe 20
```
Expected accuracy: `94.68`

```sh
python test.py --gpu 0 --S 800 --L 2 --proto deploy_resnet101_normpython.prototxt --weights model.caffemodel --dataset datasets/Paris --eval_binary datasets/evaluation/compute_ap --temp_dir tmp --dataset_name Paris –multires --aqe 1 --dbe 20
```
Expected accuracy: `96.58`

### Citation

If you use these models in your research, please cite:

```
@inproceedings{Gordo2016a,
      title={Deep Image Retrieval: Learning global representations for image search},
      author={Albert Gordo and Jon Almazan and Jerome Revaud and Diane Larlus},
      booktitke={ECCV},
      year={2016}
}   
@article{Gordo2016b,
      title={End-to-end Learning of Deep Visual Representations for Image Retrieval}
      author={Albert Gordo and Jon Almazan and Jerome Revaud and Diane Larlus},
      journal={CoRR abs/1610.07940},
      year={2016}
}
```

Please see `LICENSE.txt` for the license information.
