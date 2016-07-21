#!/usr/bin/env sh
# Compute the mean image from the training lmdb
# the caffe builds/tools directory needs to be added to the global path so
# the caffe commands can be executed


LMDB=image_data/MPIIHumanPose_val_lmdb
CAFFE_TOOLS=/home/sam/builds/caffe/build/tools
DATA=image_data/MPIIHumanPose_val_lmdb

compute_image_mean $LMDB \
  $DATA/imagenet_mean.binaryproto

echo "Done."
