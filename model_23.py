# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base_UNET import *
from model_profiler import model_profiler
from keras_applications import get_submodules_from_kwargs
from _common_blocks import Conv2dBn
from backbones_factory import Backbones
from _utils import freeze_model, filter_keras_submodules

import tensorflow as tf

def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    #concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    concat_axis = 3

    def wrapper(input_tensor, skip=None):
        x = tf.keras.layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = tf.keras.layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x

    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    #concat_axis = bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    concat_axis = bn_axis = 3

    def layer(input_tensor, skip=None):

        x = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = tf.keras.layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            x = tf.keras.layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer

def batchnorm_relu(input):

    h = tf.keras.layers.BatchNormalization()(input)
    h = tf.keras.layers.ReLU()(h)

    return h

def modified_Unet_PP(input_shape=(384, 384, 3), nclasses=1, **kwargs):

    decoder_filters=(256, 128, 64, 32, 16)
    # h_1 --> object; h_2 --> crop and weed;

    h = inputs = tf.keras.Input(input_shape)
    
    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)

    #if decoder_block_type == 'upsampling':
    #    decoder_block = DecoderUpsamplingX2Block
    #elif decoder_block_type == 'transpose':
    #    decoder_block = DecoderTransposeX2Block
    #else:
    #    raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
    #                     'Got: {}'.format(decoder_block_type))

    backbone = Backbones.get_backbone(
        'vgg16',
        #input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        input_tensor=h,
    )

    encoder_features = Backbones.get_feature_layers('vgg16', n=4)
    
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in encoder_features])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], tf.keras.layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, True, name='center_block1')(x)
        x = Conv3x3BnReLU(512, True, name='center_block2')(x)

    # building decoder blocks
    for i in range(5):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = DecoderTransposeX2Block(decoder_filters[i], stage=i, use_batchnorm=True)(x, skip)

    # model head (define number of output classes)
    x = tf.keras.layers.Conv2D(
        filters=nclasses,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)

    object_model = tf.keras.Model(input_, x)

    h = tf.nn.sigmoid(object_model.output) * h
    h_1 = tf.image.rgb_to_grayscale(h)

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

    return tf.keras.Model(inputs=input_, outputs=[object_model.output, h_1_output, h_2_output, h_3_output])
