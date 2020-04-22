# -*- coding: utf-8 -*-
from easydict import EasyDict as edict


__C                             = edict()
# Consumers can get config by: from config import cfg
cfg                             = __C

# YOLO section
__C.MODEL                       = edict()
__C.MODEL.EPOCHS                = 30
__C.MODEL.BATCH_SIZE            = 16
__C.MODEL.INP_HEIGHT            = 299
__C.MODEL.INP_WIDTH             = 299
__C.MODEL.CHANELS               = 3
__C.MODEL.SAVE_DIR              = "saved_model/"
__C.MODEL.SAVE_FREQ             = 5
__C.MODEL.TRAIN_DIR             = "dataset/train"
__C.MODEL.VALID_DIR             = "dataset/valid"
__C.MODEL.TEST_DIR              = "dataset/test"
__C.MODEL.TFRCORD_TRAIN         = "dataset/train.tfrecord"
__C.MODEL.TFRCORD_VALID         = "dataset/valid.tfrecord"
__C.MODEL.TFRCORD_TEST          = "dataset/test.tfrecord"
__C.MODEL.NUM_CLASSES           = 20

# VALID_SET_RATIO = 1 - TRAIN_SET_RATIO - TEST_SET_RATIO
TRAIN_SET_RATIO = 0.6
TEST_SET_RATIO = 0.2
