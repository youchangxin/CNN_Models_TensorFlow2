# -*- coding: utf-8 -*-
import tensorflow as tf
import math
from configuration import cfg


def swish(x):
    return x * tf.nn.sigmoid(x)


# 保证filter的大小可以被8整除
def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = None
    min_depth = min_depth or depth_divisor
    filters = filters * multiplier
    new_filter = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filter < 0.9 * filters:
        new_filter += depth_divisor
    return int(new_filter)


def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, ratio=0.25):
        super(SEBlock, self).__init__()
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reduce_conv = tf.keras.layers.Conv2D(filters=self.num_reduced_filters,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding='same')
        self.expand_conv = tf.keras.layers.Conv2D(filters=input_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding='same')

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = self.reduce_conv(branch)
        branch = swish(branch)
        branch = self.expand_conv(branch)
        branch = tf.nn.sigmoid(branch)
        output = inputs * branch
        return output


class MBConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.conv1 = tf.keras.layers.Conv2D(filters=in_channels * expansion_factor,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                         strides=stride,
                                                         padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se = SEBlock(input_channels=in_channels * expansion_factor)
        self.pointwise = tf.keras.layers.Conv2D(filters=out_channels,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.drop = tf.keras.layers.Dropout(rate=drop_connect_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = swish(x)
        x = self.depthwise(x)
        x = self.bn2(x, training=training)
        x = self.se(x)
        x = swish(x)
        x = self.pointwise(x)
        x = self.bn3(x, training=training)
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate:
                x = self.drop(x, training=training)
            x = tf.keras.layers.add([x, inputs])
        return x




