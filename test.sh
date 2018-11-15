#!/bin/bash

# baseline: run the test on Oxford dataset with single resolution

# source ~/anaconda3/bin/activate caffe

# config
PROTO=/home/processyuan/NetworkOptimization/deep-retrieval/proto/deploy_resnet101.prototxt
MODEL=/home/processyuan/NetworkOptimization/deep-retrieval/caffemodel/deep_image_retrieval_model.caffemodel
DATASET=/home/processyuan/data/Oxford/
EVAL_BINARY=/home/processyuan/NetworkOptimization/deep-retrieval/eval/compute_ap
TEMP_DIR=/home/processyuan/NetworkOptimization/deep-retrieval/eval/eval_singleres

# run
python test.py \
	--gpu 0 \
	--S 512 \
	--L 2 \
	--proto ${PROTO} \
	--weights ${MODEL} \
	--dataset ${DATASET} \
	--dataset_name Oxford \
	--eval_binary ${EVAL_BINARY} \
	--temp_dir ${TEMP_DIR} 
