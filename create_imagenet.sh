#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=image_data
DATA=image_data
TOOLS=$CAFFE_ROOT/build/tools

DATA_ROOT=$DATA/images/

echo $DATA_ROOT

# SetimagesZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: DATA_ROOT is not a path to a directory: $DATA_ROOT"
  echo "Set the DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: DATA_ROOT is not a path to a directory: $DATA_ROOT"
  echo "Set the DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $DATA/train_lst.txt \
    $EXAMPLE/MPIIHumanPose_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $DATA/val_lst.txt \
    $EXAMPLE/MPIIHumanPose_val_lmdb

echo "Done."
