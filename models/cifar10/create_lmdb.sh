#!/usr/bin/env sh
# Create the gravity images lmdb inputs

ROOT=.
LIST_ROOT=/home/jyh/datasets/cifar10/devkit
DATA_ROOT=/home/jyh/datasets/cifar10/imgs

TOOLS=/home/jyh/caffe/build/tools

rm -rf $ROOT/train_lmdb $ROOT/test_lmdb

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --shuffle \
    $DATA_ROOT/ \
    $LIST_ROOT/train.txt \
    $ROOT/train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --shuffle \
    $DATA_ROOT/ \
    $LIST_ROOT/test.txt \
    $ROOT/test_lmdb

#echo "Computing image mean..."

#$TOOLS/compute_image_mean -backend=lmdb \
#  $ROOT/train_lmdb $ROOT/mean.binaryproto


echo "Done."
