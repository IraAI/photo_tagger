#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

LMDB=/home/sam/photo_taggers/image_data/MPIIHumanPose_val_lmdb
CAFFE_TOOLS=/home/sam/builds/caffe/build/tools
DATA=/home/sam/photo_taggers/image_data/MPIIHumanPose_val_lmdb

$CAFFE_TOOLS/compute_image_mean $LMDB \
  $DATA/imagenet_mean.binaryproto

echo "Done."
