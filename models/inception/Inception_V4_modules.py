# -*- coding: utf-8 -*-
import tensorflow as tf


class BasicConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding):
        super().__init__()
        self.conv2d = tf.keras.layers.Conv2D(filters=filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv2d(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x


class Conv2dLinear(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding):
        super().__init__()
        self.conv2d = tf.keras.layers.Conv2D(filters=filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv2d(inputs)
        x = self.bn(x)
        return x


class Stem(tf.keras.layers.Layer):
    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = BasicConv2D(filters=32,
                                 kernel_size=(3, 3),
                                 strides=2,
                                 padding='valid')
        self.conv2 = BasicConv2D(filters=32,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='valid')
        self.conv3 = BasicConv2D(filters=64,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same')
        self.b1_maxpool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                    strides=2,
                                                    padding="valid")
        self.b2_conv = BasicConv2D(filters=96,
                                   kernel_size=(3, 3),
                                   strides=2,
                                   padding='valid')
        self.b3_conv1 = BasicConv2D(filters=64,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding='same')
        self.b3_conv2 = BasicConv2D(filters=96,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding='valid')
        self.b4_conv1 = BasicConv2D(filters=64,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding='same')
        self.b4_conv2 = BasicConv2D(filters=64,
                                    kernel_size=(7, 1),
                                    strides=1,
                                    padding='same')
        self.b4_conv3 = BasicConv2D(filters=64,
                                    kernel_size=(1, 7),
                                    strides=1,
                                    padding='same')
        self.b4_conv4 = BasicConv2D(filters=96,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding='valid')
        self.b5_conv = BasicConv2D(filters=192,
                                   kernel_size=(3, 3),
                                   strides=2,
                                   padding='valid')
        self.b6_maxpool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                    strides=2,
                                                    padding='valid')

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        branch_1 = self.b1_maxpool(x)
        branch_2 = self.b2_conv(x, training=training)
        x = tf.concat(values=[branch_1, branch_2], axis=-1)
        branch_3 = self.b3_conv1(x, training=training)
        branch_3 = self.b3_conv2(branch_3, training=training)
        branch_4 = self.b4_conv1(x, training)
        branch_4 = self.b4_conv2(branch_4, training=training)
        branch_4 = self.b4_conv3(branch_4, training=training)
        branch_4 = self.b4_conv4(branch_4, training=training)
        x = tf.concat(values=[branch_3, branch_4], axis=-1)
        branch_5 = self.b5_conv(x, training=training)
        branch_6 = self.b6_maxpool(x, training=training)
        x = tf.concat(values=[branch_5, branch_6], axis=-1)
        return x


class Inception_A(tf.keras.layers.Layer):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.b1_pooling = tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                           strides=1,
                                                           padding='same')
        self.b1_conv = BasicConv2D(filters=96,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding='same')
        self.b2_conv = BasicConv2D(filters=96,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding='same')
        self.b3_conv1 = BasicConv2D(filters=64,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding='same')
        self.b3_conv2 = BasicConv2D(filters=96,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding='same')
        self.b4_conv1 = BasicConv2D(filters=64,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding='same')
        self.b4_conv2 = BasicConv2D(filters=96,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding='same')
        self.b4_conv3 = BasicConv2D(filters=96,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding='same')

    def call(self, inputs, training=None, **kwargs):
        branch_1 = self.b1_pooling(inputs)
        branch_1 = self.b1_conv(branch_1, training=training)
        branch_2 = self.b2_conv(inputs, training=training)
        branch_3 = self.b3_conv1(inputs, training=training)
        branch_3 = self.b3_conv2(branch_3, training=training)
        branch_4 = self.b4_conv1(inputs, training=training)
        branch_4 = self.b4_conv2(branch_4, training=training)
        branch_4 = self.b4_conv3(branch_4, training=training)
        x = tf.concat(values=[branch_1, branch_2, branch_3, branch_4], axis=-1)
        return x


class Reduction_A(tf.keras.layers.Layer):
    def __init__(self, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.b1_maxpool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                    strides=2,
                                                    padding='valid')
        self.b2_conv = BasicConv2D(filters=n,
                                   kernel_size=(3, 3),
                                   strides=2,
                                   padding='valid')
        self.b3_conv1 = BasicConv2D(filters=k,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding='same')
        self.b3_conv2 = BasicConv2D(filters=l,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding='same')
        self.b3_conv3 = BasicConv2D(filters=m,
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding='valid')

    def call(self, inputs, training=None, **kwargs):
        branch_1 = self.b1_maxpool(inputs)
        branch_2 = self.b2_conv(inputs, training=training)
        branch_3 = self.b3_conv1(inputs, training=training)
        branch_3 = self.b3_conv2(branch_3, training=training)
        branch_3 = self.b3_conv3(branch_3, training=training)

        return tf.concat(values=[branch_1, branch_2, branch_3], axis=-1)


class Inception_B(tf.keras.layers.Layer):
    def __init__(self):
        super(Inception_B, self).__init__()
        self.b1_pooling = tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                           strides=1,
                                                           padding='same')
        self.b1_conv = BasicConv2D(filters=128,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding='same')
        self.b2_conv = BasicConv2D(filters=384,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding='same')
        self.b3_conv1 = BasicConv2D(filters=192,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding='same')
        self.b3_conv2 = BasicConv2D(filters=224,
                                    kernel_size=(1, 7),
                                    strides=1,
                                    padding='same')
        self.b3_conv3 = BasicConv2D(filters=256,
                                    kernel_size=(1, 7),
                                    strides=1,
                                    padding='same')
        self.b4_conv1 = BasicConv2D(filters=192,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding='same')
        self.b4_conv2 = BasicConv2D(filters=192,
                                    kernel_size=(1, 7),
                                    strides=1,
                                    padding='same')
        self.b4_conv3 = BasicConv2D(filters=224,
                                    kernel_size=(7, 1),
                                    strides=1,
                                    padding='same')
        self.b4_conv4 = BasicConv2D(filters=224,
                                    kernel_size=(1, 7),
                                    strides=1,
                                    padding='same')
        self.b4_conv5 = BasicConv2D(filters=256,
                                    kernel_size=(7, 1),
                                    strides=1,
                                    padding='same')

    def call(self, inputs, training=None, **kwargs):
        branch_1 = self.b1_pooling(inputs)
        branch_1 = self.b1_conv(branch_1, training=training)

        branch_2 = self.b2_conv(inputs, training=training)

        branch_3 = self.b3_conv1(inputs, training=training)
        branch_3 = self.b3_conv2(branch_3, training=training)
        branch_3 = self.b3_conv3(branch_3, training=training)

        branch_4 = self.b4_conv1(inputs, training=training)
        branch_4 = self.b4_conv2(branch_4, training=training)
        branch_4 = self.b4_conv3(branch_4, training=training)
        branch_4 = self.b4_conv4(branch_4, training=training)
        branch_4 = self.b4_conv5(branch_4, training=training)

        x = tf.concat(values=[branch_1, branch_2, branch_3, branch_4], axis=-1)
        return x


class Reduction_B(tf.keras.layers.Layer):
    def __init__(self):
        super(Reduction_B, self).__init__()
        self.b1_maxpool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                    strides=2,
                                                    padding='valid')
        self.b2_conv1 = BasicConv2D(filters=192,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding='same')
        self.b2_conv2 = BasicConv2D(filters=192,
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding='valid')
        self.b3_conv1 = BasicConv2D(filters=256,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding='same')
        self.b3_conv2 = BasicConv2D(filters=256,
                                    kernel_size=(1, 7),
                                    strides=1,
                                    padding='same')
        self.b3_conv3 = BasicConv2D(filters=320,
                                    kernel_size=(7, 1),
                                    strides=1,
                                    padding='same')
        self.b3_conv4 = BasicConv2D(filters=320,
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding='valid')

    def call(self, inputs, training=None, **kwargs):
        branch_1 = self.b1_maxpool(inputs)

        branch_2 = self.b2_conv1(inputs, training=training)
        branch_2 = self.b2_conv2(branch_2, training=training)

        branch_3 = self.b3_conv1(inputs, training=training)
        branch_3 = self.b3_conv2(branch_3, training=training)
        branch_3 = self.b3_conv3(branch_3, training=training)
        branch_3 = self.b3_conv4(branch_3, training=training)

        return tf.concat(values=[branch_1, branch_2, branch_3], axis=-1)


class Inception_C(tf.keras.layers.Layer):
    def __init__(self):
        super(Inception_C, self).__init__()
        self.b1_pooling = tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                           strides=1,
                                                           padding='same')
        self.b1_conv = BasicConv2D(filters=256,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding='same')
        self.b2_conv = BasicConv2D(filters=256,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding='same')
        self.b3_conv1 = BasicConv2D(filters=384,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding='same')
        self.b3_conv2 = BasicConv2D(filters=256,
                                    kernel_size=(1, 3),
                                    strides=1,
                                    padding='same')
        self.b3_conv3 = BasicConv2D(filters=256,
                                    kernel_size=(3, 1),
                                    strides=1,
                                    padding='same')
        self.b4_conv1 = BasicConv2D(filters=384,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding='same')
        self.b4_conv2 = BasicConv2D(filters=448,
                                    kernel_size=(1, 3),
                                    strides=1,
                                    padding='same')
        self.b4_conv3 = BasicConv2D(filters=512,
                                    kernel_size=(3, 1),
                                    strides=1,
                                    padding='same')
        self.b4_conv4 = BasicConv2D(filters=256,
                                    kernel_size=(3, 1),
                                    strides=1,
                                    padding='same')
        self.b4_conv5 = BasicConv2D(filters=256,
                                    kernel_size=(1, 31),
                                    strides=1,
                                    padding='same')

    def call(self, inputs, training=None, **kwargs):
        branch_1 = self.b1_pooling(inputs)
        branch_1 = self.b1_conv(branch_1, training=training)

        branch_2 = self.b2_conv(inputs, training=training)

        branch_3 = self.b3_conv1(inputs, training=training)
        branch_3_1 = self.b3_conv2(branch_3, training=training)
        branch_3_2 = self.b3_conv3(branch_3, training=training)

        branch_4 = self.b4_conv1(inputs, training=training)
        branch_4 = self.b4_conv2(branch_4, training=training)
        branch_4 = self.b4_conv3(branch_4, training=training)
        branch_4_1 = self.b4_conv4(branch_4, training=training)
        branch_4_2 = self.b4_conv5(branch_4, training=training)

        x = tf.concat(values=[branch_1, branch_2, branch_3_1, branch_3_2, branch_4_1, branch_4_2], axis=-1)
        return x
