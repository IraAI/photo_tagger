#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

# Set system variable for caffe root folder:
# export $CAFFE_ROOT=<caffe_root_folder>

EXAMPLE=image_data
DATA=image_data
TOOLS=$CAFFE_ROOT/build/tools

echo $DATA
echo "Starting..."

$TOOLS/compute_image_mean $EXAMPLE/MPIIHumanPose_train_lmdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
