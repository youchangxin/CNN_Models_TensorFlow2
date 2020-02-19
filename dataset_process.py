# -*- coding: utf-8 -*-
import tensorflow as tf
import pathlib
import os
import numpy as np

from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, BATCH_SIZE, \
    train_dir, test_dir, valid_dir, train_tfrecord, valid_tfrecord


def rawdata_to_tfrecord(dataset_dir, tfrecord_path):
    childirs = os.listdir(dataset_dir)
    num_categories = list(enumerate(childirs))
    filenames = []
    labels = []
    for num, category in num_categories:
        filename = [os.path.join(dataset_dir, category, i) for i in os.listdir(os.path.join(dataset_dir, category))]
        label = [num] * len(filename)
        filenames.extend(filename)
        labels.extend(label)

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for filename, label in zip(filenames, labels):
            image = open(filename, 'rb').read()  # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def _parse_example(example_string):
    # define structure of Feature
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    # decode the TFRecord file
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'], channels=CHANNELS)  # 解码JPEG图片
    return feature_dict['image'], feature_dict['label']


def _decode_and_resize(image, label):
#    image_strings = tf.io.read_file(filename)
#    image_decoded = tf.io.decode_jpeg(image_strings, channels=CHANNELS)
    image_resized = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image_norm = image_resized / 255.0
    return image_norm, label


def input_pipeline(tfrecord_path):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = raw_dataset.map(_parse_example)

    dataset = dataset.map(map_func=_decode_and_resize,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=5000).batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == '__main__':
    if not os.path.exists(train_tfrecord):
        rawdata_to_tfrecord(dataset_dir=train_dir, tfrecord_path=train_tfrecord)
        rawdata_to_tfrecord(dataset_dir=valid_dir, tfrecord_path=valid_tfrecord)

    '''
    import matplotlib.pyplot as plt

    for images, labels in dataset:
        print(images.shape)
        print(labels)

        fig, axs = plt.subplots(1, 8)
        for i in range(8):
            axs[i].set_title(labels.numpy()[i])
            axs[i].imshow(images.numpy()[i, :, :, :])
        plt.show()
    '''