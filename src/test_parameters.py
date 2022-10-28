
""" Parameters used for model evaluation """

TEST_RESULTS_DIR = "../predictions/"              				# directory where predictions are saved

TEST_MODEL = "unet/unet-amrel-processed-grismouton"				# name of the model to evaluate
#TEST_MODEL = "ddcm/ddcm-amrel-processed-grismouton"			
#TEST_MODEL = "ddcm/ddcm-cbam-dc-grismouton"			
#TEST_MODEL = "ddcm/ddcm-cbam-grismouton"			
#TEST_MODEL = "ddcm/ddcm-roads-5px-grismouton"			
TEST_AREA = "grismouton"										# area to test the model on 
TEST_INPUT_TYPES = ["slope"]									# image types to use as input during evaluation (N types will be concatenated into a single N channels image)
TEST_MASK_TYPES = ["roads-5px", "roads-1px", "roads-28px"]		# ground truth used to compute metrics : 
																#	- 1st one is used to compute all metrics if USE_AMREL_METRICS is False
																#	- 2nd one is used to compute recall if USE_AMREL_METRICS is True)
																#	- 3rd one is used to compute precision if USE_AMREL_METRICS is True)

PREDICTION_ONLY = False											# relevant if no ground truth is available for the test area : if true, the metrics aren't computed and only the predicted binary mask will be saved
SAVE_PREDICTION = True											# whether or not to save predictions as images
TEST_CITY_MASK = True 											# whether or not the binary masks hiding urban areas are applied before the evaluation
USE_AMREL_METRICS = True										# if true, the metrics are measured following AMREL's paper methodology


