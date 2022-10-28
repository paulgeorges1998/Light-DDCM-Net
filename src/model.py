"""
    Copyright 2021 Paul Georges, Phuc Ngo and Philippe Even
      authors of paper:
      Georges, P., Ngo, P. and Even, P. 2022,
      Automatic forest road extraction from LiDAR data using a convolutional
      neural network (submitted to the proceedings of the 4th workshop
      on Reproducible Research in Pattern Recognition).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import sys
import os
import pickle 

from keras import backend as K
from shutil import copyfile
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from train_parameters import *
from tensorflow.keras.callbacks import ModelCheckpoint



############################################
###########    LOSS FUNCTIONS    ###########
############################################


# Binary Cross Entropy
def create_binary_crossentropy():

    def binary_crossentropy(y_true, y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        losses = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        return K.mean(losses, axis=-1)

    return binary_crossentropy


def compute_weight(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return counts[0]/counts[1]


# Weighted Binary Cross Entropy
def create_weighted_binary_crossentropy(w):

    def weighted_binary_crossentropy(y_true, y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        losses = -(w * y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        return K.mean(losses, axis=-1)

    return weighted_binary_crossentropy


# Focal Loss (weighted)
def create_focal_loss(w):

    def focal_loss(y_true, y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        losses = -(w * y_true * ((1-y_pred) ** GAMMA) *  K.log(y_pred) + (1 - y_true) * (y_pred ** GAMMA) * K.log(1 - y_pred))
        return K.mean(losses, axis=-1)
    return focal_loss



############################################
##############    METRICS    ###############
############################################



def F1Score(precision, recall):
    """
    Custom function to compute F1-Score from keras' precision & recall functions
    """
    def F1(y_true, y_pred):
        prec = precision(y_true, y_pred)
        rec = recall(y_true, y_pred)
        score = 2 * (prec * rec) / (prec + rec + K.epsilon())
        return score

    return F1


def get_metrics():

    """

    Instantiate metrics : Binary Accuracy, Precision, Recall, F1-Score
    Default threshold is set to 0.5

    """

    threshold = 0.5
    accuracy = tf.keras.metrics.BinaryAccuracy(threshold=threshold)
    precision = tf.keras.metrics.Precision(thresholds=threshold)
    recall = tf.keras.metrics.Recall(thresholds=threshold)
    f1 = F1Score(precision, recall)

    return [accuracy, precision, recall, f1]



############################################
################    UNET    ################
############################################

def build_unet():


    #dropout_rate = 0
    input_size = (PATCH_SIZE, PATCH_SIZE, N_CHANNELS)
    initializer = 'he_normal'

    # -- Encoder -- #
    # Block encoder 1
    inputs = Input(shape=input_size)
    conv_enc_1 = Conv2D(N_FILTERS, 3, activation='relu', padding='same', kernel_initializer = initializer)(inputs)
    conv_enc_1 = Conv2D(N_FILTERS, 3, activation = 'relu', padding='same', kernel_initializer = initializer)(conv_enc_1)

    # Block encoder 2
    max_pool_enc_2 = MaxPooling2D(pool_size=(2, 2))(conv_enc_1)
    conv_enc_2 = Conv2D(2*N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_2)
    conv_enc_2 = Conv2D(2*N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_2)

    # Block  encoder 3
    max_pool_enc_3 = MaxPooling2D(pool_size=(2, 2))(conv_enc_2)
    conv_enc_3 = Conv2D(4*N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_3)
    conv_enc_3 = Conv2D(4*N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_3)

    # Block  encoder 4
    max_pool_enc_4 = MaxPooling2D(pool_size=(2, 2))(conv_enc_3)
    conv_enc_4 = Conv2D(8*N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_4)
    conv_enc_4 = Conv2D(8*N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_4)

    # ----------- #
    maxpool = MaxPooling2D(pool_size=(2, 2))(conv_enc_4)
    conv = Conv2D(16*N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(maxpool)
    conv = Conv2D(16*N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv)
    # ----------- #

    # -- Decoder -- #
    # Block decoder 1
    up_dec_1 = Conv2D(8*N_FILTERS, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv))
    merge_dec_1 = concatenate([conv_enc_4, up_dec_1], axis = 3)
    conv_dec_1 = Conv2D(8*N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_1)
    conv_dec_1 = Conv2D(8*N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_1)

    # Block decoder 2
    up_dec_2 = Conv2D(4*N_FILTERS, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_1))
    merge_dec_2 = concatenate([conv_enc_3, up_dec_2], axis = 3)
    conv_dec_2 = Conv2D(4*N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_2)
    conv_dec_2 = Conv2D(4*N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_2)

    # Block decoder 3
    up_dec_3 = Conv2D(2*N_FILTERS, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_2))
    merge_dec_3 = concatenate([conv_enc_2, up_dec_3], axis = 3)
    conv_dec_3 = Conv2D(2*N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_3)
    conv_dec_3 = Conv2D(2*N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_3)

    # Block decoder 4
    up_dec_4 = Conv2D(N_FILTERS, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_3))
    merge_dec_4 = concatenate([conv_enc_1, up_dec_4], axis = 3)
    conv_dec_4 = Conv2D(N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_4)
    conv_dec_4 = Conv2D(N_FILTERS, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
    conv_dec_4 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
    
    # -- Dencoder -- #
    output = Conv2D(1, 1, activation = 'sigmoid')(conv_dec_4)

    model = tf.keras.Model(inputs = inputs, outputs = output)

    return model



############################################
################    CBAM    ################
############################################


def attach_attention_module(net, attention_module):
  if attention_module == 'se_block': # SE_block
    net = se_block(net)
  elif attention_module == 'cbam_block': # CBAM_block
    net = cbam_block(net)
  else:
    raise Exception("'{}' is not supported attention module!".format(attention_module))

  return net

def se_block(input_feature, ratio=1):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1,1,channel//ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature

def cbam_block(cbam_feature, ratio=1):
    """Implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=1):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 3
    
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat) 
    assert cbam_feature.shape[-1] == 1
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
    return multiply([input_feature, cbam_feature])
        
    

############################################
################    DDCM    ################
############################################


def DDCM_scheduler(epoch, lr):
    """

    Learning rate decay scheduler

    """
    if epoch in [5, 15, 25, 65, 100]:
        #if epoch in [100, 150, 200]:
        return lr * 0.5
    return lr


def DDCM_block(inputs, out_dim, rates, bias=False):
    
    """

    Basic DDCM block

    """

    x = inputs
    kernel = 3

    for i, rate in enumerate(rates):
        dil_conv = Conv2D(out_dim, 3, activation='relu', padding='same', dilation_rate=rate, kernel_initializer = 'he_normal')(x)
        prelu = PReLU()(dil_conv)
        batchnorm = BatchNormalization()(prelu)

        if USE_CBAM:
            batchnorm = cbam_block(batchnorm)

        # concatenate previous feature map and new one
        x = Concatenate(axis=-1)([x, batchnorm])

    # merge all stacked features
    conv = Conv2D(out_dim, 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(x)
    prelu = PReLU()(conv)
    batchnorm = BatchNormalization()(prelu)

    return batchnorm




def build_DDCM():

    """ 
    Assemble layers to create the binary segmentation branch of the DDCM model
    """

    input_size = (PATCH_SIZE, PATCH_SIZE, N_CHANNELS)
    initializer = 'he_normal'

    inputs = Input(shape=input_size)

    # convolution to create a 2nd and/or 3rd input channel as described in DDCM paper
    channel_3 = Conv2D(3-N_CHANNELS, 3, activation='relu', padding='same', kernel_initializer = initializer)(inputs)
    concat = Concatenate(axis=-1)([inputs, channel_3])
    batchnorm = BatchNormalization()(concat)
    prelu = PReLU()(batchnorm)

    # load ResNet pretrained on ImageNet 
    resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=(PATCH_SIZE, PATCH_SIZE, 3))

    # remove last bottleneck layer
    resnet = tf.keras.Model(name='resnet_DDCM', inputs=resnet.input, outputs=resnet.layers[142].output)

    res = resnet(prelu)

    if USE_CBAM:
        res = cbam_block(res)

    # lower branch of the DDCM architecture (for binary segmentation)
    ddcm1 = DDCM_block(res, 36, [1, 2, 3, 4])

    if USE_CBAM:
        ddcm1 = cbam_block(ddcm1)

    up1 = UpSampling2D(size=4, interpolation='bilinear')(ddcm1)
    ddcm2 = DDCM_block(up1, 18, [1])

    if USE_CBAM:
        ddcm2 = cbam_block(ddcm2)

    up2 = UpSampling2D(size=2, interpolation='bilinear')(ddcm2)

    conv = Conv2D(1, 1, activation='sigmoid')(up2)
    up3 = UpSampling2D(size=2, interpolation='bilinear')(conv)

    # assemble model
    model = tf.keras.Model(inputs=inputs, outputs=up3)

    return model




#############################################
################    MODEL    ################
#############################################


def build_model(w=None):

    """

    Instantiate model, define metrics, loss function, optimizer

    """

    # assemble architecture
    if MODEL == "unet":
        model = build_unet()
    elif MODEL == "ddcm":
        model = build_DDCM()


    # instantiate metrics
    metrics = get_metrics()


    save_path = "../models/" + MODEL + "/" + NAME + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # instantiate loss function 
    if LOSS == "bce":
        loss_function = create_binary_crossentropy()

    if LOSS == "wbce": 
        if w == None:
            w = WEIGHT
        loss_function = create_weighted_binary_crossentropy(w)
        with open(save_path + "weight.pkl", 'wb+') as f:
            pickle.dump(w, f)

    if LOSS == "focal":
        if w == None:
            w = WEIGHT
        loss_function = create_focal_loss(w)
        with open(save_path + "weight.pkl", 'wb+') as f:
            pickle.dump(w, f)

    print("w=" + str(w))

    # optimizer
    if MODEL == "unet":
        optimizer = Adam(learning_rate=LEARNING_RATE)
    elif MODEL == "ddcm":
        optimizer = tfa.optimizers.AdamW(weight_decay=0.00005, learning_rate=LEARNING_RATE, amsgrad=True)


    # compile model
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics, run_eagerly=False)


    # callbacks to monitor training and save best model
    checkpoint = ModelCheckpoint(save_path + "/" + NAME + ".hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto', save_freq="epoch")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=save_path + "/logs/", histogram_freq=1)
    callbacks = [checkpoint, tensorboard_callback]

    # add callback for lr scheduler if DDCM is used
    if MODEL == "ddcm":
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(DDCM_scheduler)
        callbacks.append(lr_scheduler)

    # save current parameters 
    copyfile("train_parameters.py", save_path + "train_parameters.py")


    # save architecture as an image
    tf.keras.utils.plot_model(model, to_file="../models/" + MODEL + ".png", show_shapes=True)


    return model, loss_function, metrics, callbacks



def load_trained_model(model_name):

    """
    Load saved model's architecture & weights
    """

    # instantiate loss function 
    if LOSS == "bce":
        loss_function = create_binary_crossentropy()

    if LOSS == "wbce" :
        with open("../models/" + MODEL + "/" + NAME + "/weight.pkl", "rb") as f:
            w = pickle.load(f)
            loss_function = create_weighted_binary_crossentropy(w)


    if LOSS == "focal" :
        with open("../models/" + MODEL + "/" + NAME + "/weight.pkl", "rb") as f:
            w = pickle.load(f)
            loss_function = create_focal_loss(w)


    # instantiate metrics
    acc, prec, rec, F1 = get_metrics()

    # load trained model
    archi, name = model_name.split("/")
    model = tf.keras.models.load_model("../models/" + archi + "/" + name + ".hdf5", custom_objects={"bce":loss_function,"weighted_binary_crossentropy":loss_function, "focal_loss":loss_function, "F1":F1})


    return model
