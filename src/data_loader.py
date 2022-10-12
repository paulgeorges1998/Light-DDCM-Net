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
import numpy as np
import sys
import random
import cv2
import os

from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from train_parameters import *
from test_parameters import *




def split_into_patches(images, area, types):

    """ 

    Split images into non-overlapping patches

    """

    channels = tf.shape(images)[3]
    batch_size = tf.shape(images)[0]

    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
        strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
        rates=[1, 1, 1, 1],
        padding="SAME"
    )

    patches_shape = patches.shape

    # save patches if param is true
    if SAVE_PATCHES:

        print("Saving patches for " + area + " area ... ")

        path = DATA_DIR + area + "/" + str(PATCH_SIZE) + "x" + str(PATCH_SIZE) + "/" 
        if not os.path.exists(path):
            os.makedirs(path)

        patches = tf.reshape(patches, [patches_shape[0], patches_shape[1], patches_shape[2], PATCH_SIZE, PATCH_SIZE, tf.shape(images)[3]])
        patches = tf.squeeze(patches, axis=0)

        for j in range(patches.shape[0]):
            for i in range(patches.shape[1]):
                patch = patches[j][i]
                patch = tf.experimental.numpy.moveaxis(patch, 2, 0)
                for k in range(len(types)):
                    cv2.imwrite(str(path + area + "_" + types[k] + "_" + str(j).zfill(3) + "-" + str(i).zfill(3) + ".png"), np.float32(patch[k]))

    patches = tf.reshape(patches, [patches_shape[1]*patches_shape[2], PATCH_SIZE, PATCH_SIZE, tf.shape(images)[3]])

    return patches  # (N_PATCHES, PATCH_SIZE, PATCH_SIZE, CHANNELS)





def load_data(areas, input_types, mask_types, shuffle_b, hide_city):

    """ 

    Load desired input images and ground truth masks for a list of areas 
    
    The images will be split into square patches of dimensions PATCH_SIZE x PATCH_SIZE
    
    If there are more than one type of inputs/masks, the N > 1 types are concatenated into N channels patches (PATCH_SIZE x PATCH_SIZE x N)

    """

    masks = []
    images = []
    areas_dimensions = []

    for area in areas:

        area_dimensions = (-1, -1)

        # load all matching input images
        area_images = []
        for input_type in input_types:
            image_path = DATA_DIR + area + "/" + area + "_" + input_type + ".png"
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            area_images.append(image)
            area_dimensions = image.shape

        areas_dimensions.append(area_dimensions)

        # same for all matching mask images
        area_masks = []

        for mask_type in mask_types:
            mask_path = DATA_DIR + area + "/" + area + "_" + mask_type + ".png"
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            area_masks.append(mask)

        # load and apply the mask hiding urban areas to input images (if param is true)
        if hide_city:
            mask_type = "city-mask"
            mask_path = DATA_DIR + area + "/" + area + "_" + mask_type + ".png"
            city_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            for i in range(len(area_images)):
                area_images[i] = cv2.bitwise_and(area_images[i], city_mask)
            for i in range(len(area_masks)):
                area_masks[i] = cv2.bitwise_and(area_masks[i], city_mask)

        # split original images & masks into non-overlapping patches
        area_images = np.expand_dims(np.dstack(area_images), axis=0)
        area_images_patches = split_into_patches(area_images, area, input_types)
        area_images_patches = np.array(area_images_patches, dtype="float16") / 255   # [0, 1] normalization
        images.append(area_images_patches)

        if len(mask_types) > 0:
            area_masks = np.expand_dims(np.dstack(area_masks), axis=0)
            area_masks_patches = split_into_patches(area_masks, area, mask_types)
            area_masks_patches = np.array(area_masks_patches, dtype="float16") / 255
            masks.append(area_masks_patches)



    # shuffle data if not testing
    if shuffle_b:
        images, masks = shuffle(images, masks)


    images = np.vstack(images)      # (N_PATCHES * N_AREAS, PATCH_SIZE, PATCH_SIZE, CHANNELS)
    if len(mask_types) > 0:
        masks = np.vstack(masks)    # (N_PATCHES * N_AREAS, PATCH_SIZE, PATCH_SIZE, CHANNELS)


    return images, masks, areas_dimensions



def load_train_data():
    x, y, areas_dimensions = load_data(TRAIN_AREAS, INPUT_TYPES, [MASK_TYPE], True, CITY_MASK)
    return x, y


def load_test_data():
    x, y, areas_dimensions = load_data([TEST_AREA], TEST_INPUT_TYPES, TEST_MASK_TYPES, False, TEST_CITY_MASK)
    return x, y, areas_dimensions[0]


def load_test_data_no_masks():
    x, y, areas_dimensions = load_data([TEST_AREA], TEST_INPUT_TYPES, [], False, TEST_CITY_MASK)
    return x, areas_dimensions[0]



def combine_generators(input_generators, mask_generator):

    """

    Merge a list of input generators and a mask generator into a single one that yields ([input1, input2, ...], mask) tuples


    """

    while True:
        x_list = []
        for gen in input_generators:
            x_list.append(np.squeeze(gen.next()))

        x_list = np.array(x_list)

        if x_list.ndim == 3:
            x_list = np.expand_dims(x_list, axis=1)

        y = mask_generator.next()

        # threshold to repair interpolations caused by the rotation in the mask
        y = np.where(y>0.5, 1., 0.)

        yield (tf.convert_to_tensor(np.stack(x_list, axis=3)), tf.convert_to_tensor(y))



def data_augmentation(x, y):

    """

    Returns a generator yielding ([input1, input2, ...], mask) tuples of augmented images

    """

    augment_dict = dict(
        rotation_range = ROTATION_RANGE,                                   
        horizontal_flip = HORIZONTAL_FLIP,                              
        fill_mode = FILL_MODE 
    )


    # provide the same seed to all generators in order to get coherent augmentations on images and masks
    seed = random.randint(0, 1000)

    
    # reshape x to be able to iterate over channels
    x = np.moveaxis(x, 3, 0)[..., np.newaxis]   # x : (N_PATCHES, PATCH_SIZE, PATCH_SIZE, CHANNELS) -> (CHANNELS, N_PATCHES, PATCH_SIZE, PATCH_SIZE, 1)
                                                # y : (N_PATCHES, PATCH_SIZE, PATCH_SIZE, 1)

    # create one generator for masks
    mask_datagen = ImageDataGenerator(**augment_dict)
    mask_generator = mask_datagen.flow(x=y, batch_size=BATCH_SIZE, shuffle=False, seed=seed)
    

    # create one generator for each input type
    image_generators = []
    for i in range(x.shape[0]):
        image_datagen = ImageDataGenerator(**augment_dict)
        if SAVE_PATCHES:
            image_generator = image_datagen.flow(x=x[i], batch_size=BATCH_SIZE, shuffle=False, seed=seed, save_to_dir="./images/")
        else:
            image_generator = image_datagen.flow(x=x[i], batch_size=BATCH_SIZE, shuffle=False, seed=seed)
        image_generators.append(image_generator)


    # merge generators into a single one that yields ([input1, input2, ...], mask) tuples
    train_generator = combine_generators(image_generators, mask_generator)


    return train_generator



