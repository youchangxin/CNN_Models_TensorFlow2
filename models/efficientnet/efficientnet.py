# -*- coding: utf-8 -*-
import tensorflow as tf
from models.efficientnet.efficientnet_block import MBConv, round_filters, round_repeats, swish
from configuration import cfg

NUM_CLASSES = cfg.MODEL.NUM_CLASSES


def build_mbconv_block(in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate):
    block = tf.keras.Sequential()
    for i in range(layers):
        if i == 0:
            block.add(MBConv(in_channels=in_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=stride,
                             k=k,
                             drop_connect_rate=drop_connect_rate))
        else:
            block.add(MBConv(in_channels=out_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=1,
                             k=k,
                             drop_connect_rate=drop_connect_rate))
    return block


class EfficientNet(tf.keras.Model):
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate=0.2):
        super(EfficientNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=round_filters(32, width_coefficient),
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.block1 = build_mbconv_block(in_channels=round_filters(32, width_coefficient),
                                         out_channels=round_filters(16, width_coefficient),
                                         layers=round_repeats(1, depth_coefficient),
                                         stride=1,
                                         expansion_factor=1,
                                         k=3,
                                         drop_connect_rate=drop_connect_rate)
        self.block2 = build_mbconv_block(in_channels=round_filters(16, width_coefficient),
                                         out_channels=round_filters(24, width_coefficient),
                                         layers=round_repeats(2, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)
        self.block3 = build_mbconv_block(in_channels=round_filters(24, width_coefficient),
                                         out_channels=round_filters(40, width_coefficient),
                                         layers=round_repeats(2, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
        self.block4 = build_mbconv_block(in_channels=round_filters(40, width_coefficient),
                                         out_channels=round_filters(80, width_coefficient),
                                         layers=round_repeats(3, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)
        self.block5 = build_mbconv_block(in_channels=round_filters(80, width_coefficient),
                                         out_channels=round_filters(112, width_coefficient),
                                         layers=round_repeats(3, depth_coefficient),
                                         stride=1,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
        self.block6 = build_mbconv_block(in_channels=round_filters(112, width_coefficient),
                                         out_channels=round_filters(192, width_coefficient),
                                         layers=round_repeats(4, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
        self.block7 = build_mbconv_block(in_channels=round_filters(192, width_coefficient),
                                         out_channels=round_filters(320, width_coefficient),
                                         layers=round_repeats(1, depth_coefficient),
                                         stride=1,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)

        self.conv2 = tf.keras.layers.Conv2D(filters=round_filters(1280, width_coefficient),
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = swish(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = swish(x)
        x = self.pool(x)
        x = self.dropout(x, training=training)
        x = self.fc(x)

        return x


def get_efficientnet(width_coefficient, depth_coefficient, resolution, dropout_rate):
    net = EfficientNet(width_coefficient=width_coefficient,
                       depth_coefficient=depth_coefficient,
                       dropout_rate=dropout_rate)
    return net


def efficient_net_b0():
    return get_efficientnet(1.0, 1.0, 224, 0.2)


def efficient_net_b1():
    return get_efficientnet(1.0, 1.1, 240, 0.2)


def efficient_net_b2():
    return get_efficientnet(1.1, 1.2, 260, 0.3)


def efficient_net_b3():
    return get_efficientnet(1.2, 1.4, 300, 0.3)


def efficient_net_b4():
    return get_efficientnet(1.4, 1.8, 380, 0.4)


def efficient_net_b5():
    return get_efficientnet(1.6, 2.2, 456, 0.4)


def efficient_net_b6():
    return get_efficientnet(1.8, 2.6, 528, 0.5)


def efficient_net_b7():
    return get_efficientnet(2.0, 3.1, 600, 0.5)


model = efficient_net_b5()
model.build(input_shape=(None, 456, 456, 3))
model.summary()