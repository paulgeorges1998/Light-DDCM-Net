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


import glob
import cv2
import os
import sys
import numpy as np

from train_parameters import *


"""

	Generate all ground truths from center line & processed amrel 

"""


def load_grayscale_image(path):

	# loads image, converts it to 8-bit grayscale
	im = cv2.imread(path)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# extract file name
	basename = os.path.basename(path)
	file_name = os.path.splitext(basename)[0]

	return file_name, im


# CENTER LINE 
for path in glob.iglob(DATA_DIR + "*/*_roads-1px.png"):
	
	# load center line image
	file_name, gt = load_grayscale_image(path)
	area, category = file_name.split("_")
	 
	# 5px roads
	kernel = np.ones((3, 3), np.uint8)
	gt_5px = cv2.dilate(gt, kernel, iterations=2)
	cv2.imwrite(DATA_DIR + area + "/" + area + "_roads-5px.png", gt_5px)
	print(DATA_DIR + area + "/" + area + "_roads-5px.png")

	# 28px roads
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(28,28))
	gt_28px = cv2.dilate(gt, kernel, iterations=1)
	cv2.imwrite(DATA_DIR + area + "/" + area + "_roads-28px-ellipse.png", gt_28px)
	print(DATA_DIR + area + "/" + area + "_roads-28px-ellipse.png")



# AMREL 
for path in glob.iglob(DATA_DIR + "*/*_amrel-processed.png"):

	# load amrel processed image
	file_name, gt_amrel = load_grayscale_image(path)
	area, category = file_name.split("_")

	# skeletonization
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
	dilation = cv2.dilate(gt_amrel, kernel, iterations=1)
	skeleton = cv2.ximgproc.thinning(dilation)
	cv2.imwrite(DATA_DIR + area + "/" + area + "_amrel-1px.png", skeleton)
	print(DATA_DIR + area + "/" + area + "_amrel-1px.png")


	# dilation of the skeleton (28px road)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(28,28))
	dilation = cv2.dilate(skeleton, kernel, iterations=1)
	cv2.imwrite(DATA_DIR + area + "/" + area + "_amrel-28px.png", dilation)
	print(DATA_DIR + area + "/" + area + "_amrel-28px.png")


