CAFFE_ROOT=/home/sam/builds/caffe
LABELS=/home/sam/photo_taggers/image_data
DATA=/home/sam/photo_taggers/image_data/train/
LMDB=/home/sam/photo_taggers/image_data

GLOG_logtostderr=1 $CAFFE_ROOT/build/tools/convert_imageset \
    $DATA \
    $LABELS/train_lst.txt \
    $LMDB/MPIIHumanPose_train_lmdb
    
DATA=/home/sam/photo_taggers/image_data/val/


GLOG_logtostderr=1 $CAFFE_ROOT/build/tools/convert_imageset \
    $DATA \
    $LABELS/val_lst.txt \
    $LMDB/MPIIHumanPose_val_lmdb
