#!/usr/bin/env sh
set -e

TOOLS=$CAFFE_ROOT/build/tools

$TOOLS/caffe train \
  --solver=model/VGG_16_layers/vgg.solver.prototxt
  --weights=model/VGG_16_layers/vgg.caffemodel
