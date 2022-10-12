

""" Training parameters (some params such as weight decay are hard-coded in the model definition, you can change their values there) """


MODEL = "ddcm"											# model used for training {"unet", "ddcm"}
LOSS = "bce"											# loss function, binary crossentropy, weighted binary crossentropy, focal loss {bce, wbce, focal} 
WEIGHT = None 											# if wbce is used, weight for class 1. If None, computed as n_pixels_0 / n_pixels_1
NAME = "ddcm-amrel-processed-grismouton"				# what the best model saved during training will be called
DATA_DIR = "../data/"                                  	# directory containing data


MASK_TYPE = "amrel-processed"                          	# ground truth used during training
INPUT_TYPES = ["slope"]                                	# image types to use as input (N types will be concatenated into a single N channels image)
TRAIN_AREAS = ["grandrupt", "cuveaux", "stmont"]		# areas to be used as training/validation data

CITY_MASK = True 										# whether or not the binary masks hiding urban areas are applied
SAVE_PATCHES = False									# whether or not to save the patches obtained after splitting the original images
USE_CBAM = False										# whether or not to use CBAM Attention module (DDCM only)

LEARNING_RATE = 0.00012									# learning rate
EPOCHS = 100                                          	# max number of epochs
BATCH_SIZE = 8                                        	# size of a batch
PATCH_SIZE = 256                                       	# size of square patches to divide the original images into
N_CHANNELS = len(INPUT_TYPES)                          	# number of channels of the input images
N_FILTERS = 50                                         	# U-Net only : number of filters of the first convolution layer
VALIDATION_RATIO = 0.2                                 	# proportion of training data to be used as validation dataset
GAMMA = 2												# parameter of focal loss, see formula


ROTATION_RANGE = 179                                  	# 
HORIZONTAL_FLIP = True                                 	# data augmentation parameters, see tf.keras.preprocessing.image.ImageDataGenerator 
FILL_MODE = "reflect"                                  	#

