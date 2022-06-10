# -*- coding:utf-8 -*-
from model_profiler import model_profiler
import tensorflow as tf

def batchnorm_relu(input):

    h = tf.keras.layers.BatchNormalization()(input)
    h = tf.keras.layers.ReLU()(h)

    return h

def modified_Unet_PP(input_shape=(384, 384, 3), nclasses=1):
    # h_1 --> object; h_2 --> crop and weed;
    h = inputs = tf.keras.Input(input_shape)
    h_1 = tf.image.rgb_to_grayscale(h)
    encoder_backbone = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False)

    #
    h_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_1 = tf.keras.layers.MaxPool2D((2,2), 2)(h_1)

    h_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1_att = h_1
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h)
    h_2_1 = h_2[:, :, :, 0:32]
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)
    h_2_2 = h_2[:, :, :, 32:]
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)

    h_2_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h_2_1)
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)
    h_2_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h_2_2)
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)

    h_2 = tf.concat([h_2_1, h_2_2], -1)
    h_2 = tf.keras.layers.MaxPool2D((2,2), 2)(h_2)
    h_2 = tf.nn.sigmoid(h_1_att) * h_2

    #
    h_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    block_2 = h_1

    h_1 = tf.keras.layers.MaxPool2D((2,2), 2)(h_1)

    h_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1_att = h_1
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False, groups=2)(h_2)
    h_2_1 = h_2[:, :, :, 0:64]
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)
    h_2_2 = h_2[:, :, :, 64:]
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)

    h_2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h_2_1)
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)
    block_1_2 = h_2_1
    h_2_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h_2_2)
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)
    block_2_2 = h_2_2
    
    h_2 = tf.concat([h_2_1, h_2_2], -1)
    h_2 = tf.keras.layers.MaxPool2D((2,2), 2)(h_2)
    h_2 = tf.nn.sigmoid(h_1_att) * h_2

    #
    h_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    block_3 = h_1

    h_1 = tf.keras.layers.MaxPool2D((2,2), 2)(h_1)

    h_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1_att = h_1
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False, groups=2)(h_2)
    h_2_1 = h_2[:, :, :, 0:128]
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)
    h_2_2 = h_2[:, :, :, 128:]
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)

    h_2_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_2_1)
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)
    h_2_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_2_1)
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)
    block_1_3 = h_2_1
    h_2_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_2_2)
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)
    h_2_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_2_2)
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)
    block_2_3 = h_2_2

    h_2 = tf.concat([h_2_1, h_2_2], -1)
    h_2 = tf.keras.layers.MaxPool2D((2,2), 2)(h_2)
    h_2 = tf.nn.sigmoid(h_1_att) * h_2

    #
    h_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    block_4 = h_1

    h_1 = tf.keras.layers.MaxPool2D((2,2), 2)(h_1)  #############

    h_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1_att = h_1
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False, groups=2)(h_2)
    h_2_1 = h_2[:, :, :, 0:256]
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)
    h_2_2 = h_2[:, :, :, 256:]
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)

    h_2_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_2_1)
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)
    h_2_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_2_1)
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)
    block_1_4 = h_2_1
    h_2_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_2_2)
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)
    h_2_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_2_2)
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)
    block_2_4 = h_2_2

    h_2 = tf.concat([h_2_1, h_2_2], -1)
    h_2 = tf.keras.layers.MaxPool2D((2,2), 2)(h_2)
    h_2 = tf.nn.sigmoid(h_1_att) * h_2

    #
    h_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2, use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.concat([h_1, block_4], -1)
    h_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1_att = h_1
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_2_1 = h_2[:, :, :, 0:256]
    h_2_1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2, use_bias=False)(h_2_1)
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)
    h_2_1 = tf.concat([h_2_1, block_1_4], -1)
    h_2_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_2_1)
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)

    h_2_2 = h_2[:, :, :, 256:]
    h_2_2 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2, use_bias=False)(h_2_2)
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)
    h_2_2 = tf.concat([h_2_2, block_2_4], -1)
    h_2_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_2_2)
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)

    h_2 = tf.concat([h_2_1, h_2_2], -1)
    h_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False, groups=2)(h_2)
    h_2 = tf.nn.sigmoid(h_1_att) * h_2
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    #
    h_1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.concat([h_1, block_3], -1)
    h_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1_att = h_1
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_2_1 = h_2[:, :, :, 0:128]
    h_2_1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, use_bias=False)(h_2_1)
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)
    h_2_1 = tf.concat([h_2_1, block_1_3], -1)
    h_2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h_2_1)
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)

    h_2_2 = h_2[:, :, :, 128:]
    h_2_2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, use_bias=False)(h_2_2)
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)
    h_2_2 = tf.concat([h_2_2, block_2_3], -1)
    h_2_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h_2_2)
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)

    h_2 = tf.concat([h_2_1, h_2_2], -1)
    h_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False, groups=2)(h_2)
    h_2 = tf.nn.sigmoid(h_1_att) * h_2
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    #
    h_1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.concat([h_1, block_2], -1)
    h_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1_att = h_1
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_2_1 = h_2[:, :, :, 0:64]
    h_2_1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(h_2_1)
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)
    h_2_1 = tf.concat([h_2_1, block_1_2], -1)
    h_2_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h_2_1)
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)

    h_2_2 = h_2[:, :, :, 64:]
    h_2_2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(h_2_2)
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)
    h_2_2 = tf.concat([h_2_2, block_2_2], -1)
    h_2_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h_2_2)
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)

    h_2 = tf.concat([h_2_1, h_2_2], -1)
    h_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False, groups=2)(h_2)
    h_2 = tf.nn.sigmoid(h_1_att) * h_2
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    #
    h_1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1_att = h_1
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_1_output = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=1)(h_1)

    h_2_1 = h_2[:, :, :, 0:32]
    h_2_1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(h_2_1)
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)
    h_2_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(h_2_1)
    h_2_1 = tf.nn.sigmoid(h_1_att) * h_2_1
    h_2_1 = tf.keras.layers.BatchNormalization()(h_2_1)
    h_2_1 = tf.keras.layers.ReLU()(h_2_1)

    h_2_output = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=1)(h_2_1)

    h_2_2 = h_2[:, :, :, 32:]
    h_2_2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(h_2_2)
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)
    h_2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(h_2_2)
    h_2_2 = tf.nn.sigmoid(h_1_att) * h_2_2
    h_2_2 = tf.keras.layers.BatchNormalization()(h_2_2)
    h_2_2 = tf.keras.layers.ReLU()(h_2_2)

    h_3_output = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=1)(h_2_2)

    return tf.keras.Model(inputs=inputs, outputs=[h_1_output, h_2_output, h_3_output])

