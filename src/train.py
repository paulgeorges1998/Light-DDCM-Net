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

import sys
import tensorflow as tf

from data_loader import *
from train_parameters import *
from model import *
from sklearn.model_selection import train_test_split



def train():
    

    # load input data
    x_data, y_data = load_train_data()
    y_data = np.where(y_data > 0.5, 1., 0.)
    

    # compute weights for wbce loss if used
    if (LOSS == "wbce" or LOSS == "focal") and WEIGHT == None:
        w = compute_weight(y_data)


    # split data into training and validation datasets
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, shuffle=True, test_size=VALIDATION_RATIO)


    # apply data augmentation on the training set
    # a generator is used to augment data "on the fly" during training
    train_generator = data_augmentation(x_train, y_train)


    # compute parameters
    STEPS_PER_EPOCH = (len(x_data) * (1-VALIDATION_RATIO)) // BATCH_SIZE
    VALIDATION_STEPS = (len(x_data) * VALIDATION_RATIO) // BATCH_SIZE
    

    # build model
    if LOSS == "wbce" or LOSS == "focal":
        model, loss_function, metrics, callbacks = build_model(w)
    else :
        model, loss_function, metrics, callbacks = build_model()


    # start training
    model_history = model.fit(train_generator,
                              epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_data=(x_val, y_val),
                              validation_steps=VALIDATION_STEPS,
                              validation_batch_size=BATCH_SIZE,
                              callbacks=callbacks)



train()


