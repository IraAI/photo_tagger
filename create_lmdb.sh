# the caffe builds/tools directory needs to be added to the global path so
# the caffe commands can be executed

CAFFE_ROOT=/home/sam/builds/caffe
LABELS=image_data
DATA=image_data/train/
LMDB=image_data

GLOG_logtostderr=1 convert_imageset \
    $DATA \
    $LABELS/train_lst.txt \
    $LMDB/MPIIHumanPose_train_lmdb
    
DATA=image_data/val/


GLOG_logtostderr=1 convert_imageset \
    $DATA \
    $LABELS/val_lst.txt \
    $LMDB/MPIIHumanPose_val_lmdb
