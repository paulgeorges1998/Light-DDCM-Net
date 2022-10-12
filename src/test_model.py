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
import time
import math

from model import *
from train_parameters import *
from test_parameters import *
from data_loader import *
from shutil import copyfile



"""

Evaluate model on test set, save predictions

"""


def compute_metrics(y_test, pred_masks, threshold):

    binacc = tf.keras.metrics.BinaryAccuracy(threshold=threshold)   
    recall = tf.keras.metrics.Recall(thresholds=threshold)
    precision = tf.keras.metrics.Precision(thresholds=threshold)

    # whether or not to follow AMREL's paper methodology to compute metrics
    if USE_AMREL_METRICS:
        mask, recall_mask, precision_mask = y_test
    else:
        mask, recall_mask, precision_mask = y_test[0], y_test[0], y_test[0]


    binacc.update_state(mask, pred_masks)
    recall.update_state(recall_mask, pred_masks)
    precision.update_state(precision_mask, pred_masks)

    acc = float(binacc.result())
    rec = float(recall.result())
    prec = float(precision.result())

    f1 = 2 * (prec * rec) / (prec + rec)

    prec, rec, acc, f1 = round(prec, 4), round(rec, 4), round(acc, 4), round(f1, 4)

    print("(threshold " + str(threshold) + ")")
    print("binary_accuracy = " + str(acc) + ", precision = " + str(prec) + ", recall = " + str(rec) + ", F1 = " + str(f1) + "\n")

    return acc, prec, rec, f1


def prediction(images, model):

    n_patches = images.shape[0]

    # split patches into batches
    # useful when testing large areas (predict one batch then remove it from memory)
    PRED_BATCH_SIZE = 256
    images = [images[i:i + PRED_BATCH_SIZE] for i in range(0, n_patches, PRED_BATCH_SIZE)]

    # make predictions and measure duration 
    timer = 0
    pred_masks = []
    while len(images) > 0:
        batch = images.pop(0)
        start = time.time()
        pred = model.predict(batch, batch_size=1)
        timer += time.time() - start
        pred_masks.extend(np.array(pred, dtype="float16"))

    timer = round(timer, 2)
    pred_masks = np.array(pred_masks)
    images = []

    print("\n---------------------------------------------------\n")
    print("Model tested on " + TEST_AREA + " area\n")
    print(str(n_patches) + " patches of size " + str(PATCH_SIZE) + "x" + str(PATCH_SIZE) + " predicted in " + str(timer) + "s (" +  str(round(1000*timer/n_patches, 2)) + "ms/tile)")
    print("\n---------------------------------------------------\n")

    return pred_masks, timer



# reconstruct original images from patches
def reconstruct_images(tiled_images, area_dimensions):

    ret_images = []

    height, width = area_dimensions

    # count tiles in original image
    dimX = int(math.ceil(width/PATCH_SIZE))
    dimY = int(math.ceil(height/PATCH_SIZE))

    # indexes of padded pixels on first and last row/column
    offsetX = int((dimX * PATCH_SIZE - width) / 2)
    offsetY = int((dimY * PATCH_SIZE - height) / 2)
    indexesX1 = np.s_[0:offsetX]
    indexesY1 = np.s_[0:offsetY]
    indexesX2 = np.s_[width:width + offsetX]
    indexesY2 = np.s_[height:height + offsetY]

    
    # assemble tiles 
    for tiled_image in tiled_images:

        tiled_image = np.reshape(tiled_image, (dimY, dimX, PATCH_SIZE, PATCH_SIZE))
        image = np.zeros((dimY * PATCH_SIZE, dimX * PATCH_SIZE), dtype="float16")

        for y in range(dimY): 
            for x in range(dimX):
                image[y*PATCH_SIZE:(y+1)*PATCH_SIZE, x*PATCH_SIZE:(x+1)*PATCH_SIZE] = tiled_image[y][x]

        tiled_image = []

        # remove padded pixels 
        image = np.delete(image, indexesY1, axis=0)
        image = np.delete(image, indexesX1, axis=1)
        image = np.delete(image, indexesY2, axis=0)
        image = np.delete(image, indexesX2, axis=1)

        ret_images.append(image)

    return ret_images




def save_predictions(masks, pred, area_dimensions, threshold):

    path = TEST_RESULTS_DIR + TEST_MODEL

    # apply threshold to predicted output
    pred_mask = np.where(pred > threshold, 1., 0.)

    pred, pred_mask = reconstruct_images([pred, pred_mask], area_dimensions)

    cv2.imwrite(str(path + "/prediction_raw.png"), np.float32(pred) * 255)
    cv2.imwrite(str(path + "/prediction_mask.png"), np.float32(pred_mask) * 255)

    pred_raw = []


    if not PREDICTION_ONLY:
   

        if USE_AMREL_METRICS:
            mask, recall_mask, precision_mask = reconstruct_images(masks, area_dimensions)
        else : 
            mask = reconstruct_images([masks[0]], area_dimensions)[0]
            recall_mask, precision_mask = mask, mask


        # get false positives, false negatives, and correct predictions (compared to dilated gt & skeletonized gt if used)
        # make a RGB image of results (red = FP, green = True, black = FN)
        FP = np.clip(pred_mask - precision_mask, 0, 1)
        FN = np.clip(recall_mask - pred_mask, 0, 1)
        T = np.where(pred_mask == precision_mask, pred_mask, 0) 
        inv_mask = np.clip(1 - precision_mask, 0, 1)
        R = inv_mask
        G = np.clip(inv_mask - FP + T, 0, 1)
        B = np.clip(inv_mask - FP, 0, 1)
        pred_mask_overlay = np.stack([B, G, R], axis=2)


        # get "real" false positives & false negatives (compared gt not dilated gt or skeletonized)
        FP = np.clip(pred_mask - mask, 0, 1)
        FN = np.clip(mask - pred_mask, 0, 1)

        cv2.imwrite(str(path + "/ground_truth.png"), np.float32(mask) * 255)
        cv2.imwrite(str(path + "/FP.png"),np.float32(FP) * 255)
        cv2.imwrite(str(path + "/FN.png"), np.float32(FN) * 255)   
        cv2.imwrite(str(path + "/T.png"), np.float32(T) * 255)
        cv2.imwrite(str(path + "/prediction_overlay.png"), np.float32(pred_mask_overlay) * 255)



def test_model():

    model = load_trained_model(TEST_MODEL)

    
    # load test data
    if PREDICTION_ONLY:
        # load input images only
        x_test, area_dimensions = load_test_data_no_masks()
        y_test = []
    else:
        # load input images and ground truth masks
        x_test, y_test, area_dimensions = load_test_data()
        y_test = tf.experimental.numpy.moveaxis(y_test, 3, 0)   #(CHANNELS, N_PATCHES, PATCH_SIZE, PATCH_SIZE)


    # make predictions, measure duration
    pred_masks, timer = prediction(x_test, model)
    x_test = []


    # save timer
    save_path = TEST_RESULTS_DIR + TEST_MODEL
    print("Prediction saved in " + save_path + " directory")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = open(save_path + "/results.txt", "w+")
    save_file.write("length=" + str(timer) + "s\n")


    # threshold for metrics & output binary mask
    threshold = 0.5 


    # compute & save metrics
    if not PREDICTION_ONLY:
        acc, prec, rec, f1 = compute_metrics(y_test, pred_masks, threshold)
        save_file.write("binary_accuracy=" + str(acc) + "\n")
        save_file.write("precision=" + str(prec) + "\n")
        save_file.write("recall=" + str(rec) + "\n")
        save_file.write("f1=" + str(f1) + "\n")

    save_file.close()
    
    
    # save parameters 
    copyfile("test_parameters.py", save_path + "/test_parameters.py")


    # save predicted images
    if SAVE_PREDICTION:
        save_predictions(y_test, pred_masks, area_dimensions, threshold)


    print("\n---------------------------------------------------\n")
    print("Done.\n")


test_model()


