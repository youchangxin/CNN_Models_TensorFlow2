# -*- coding: utf-8 -*-
EPOCHS = 30
BATCH_SIZE = 32
NUM_CLASSES = 2
IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299
CHANNELS = 3
save_model_dir = r"saved_model/"
save_every_n_epoch = 5
dataset_dir = "dataset"
train_dir = dataset_dir + r"\train"
valid_dir = dataset_dir + r"\valid"
test_dir = dataset_dir + r"\test"
train_tfrecord = dataset_dir + r"/train.tfrecord"
valid_tfrecord = dataset_dir + r"/valid.tfrecord"
test_tfrecord = dataset_dir + r"/test.tfrecord"
# VALID_SET_RATIO = 1 - TRAIN_SET_RATIO - TEST_SET_RATIO
TRAIN_SET_RATIO = 0.6
TEST_SET_RATIO = 0.2
