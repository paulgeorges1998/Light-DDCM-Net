
# Light-DDCM-Net

Implementation of the architectures described in the paper _Automatic forest road extraction from LiDAR data using a convolutional neural network_

The L-DDCM-Net architectures are based on the DDCM-net architecture described in _Dense Dilated Convolutions Merging Network for Semantic Mapping of Remote Sensing Images_	https://doi.org/10.48550/arXiv.1908.11799

The LiDAR data used to create the training images were acquired in the scope of PCR AGER project (Projet collectif de recherche — Archéologie et GEoarchéologie du premier Remiremont et de ses abords).
Contact: Charles Kraemer, Pôle Archéologique Universitaire - HISCANT-MA,
Universit\'e de Lorraine, 91 avenue de la Libération, 54000 NANCY, France.


## Dependencies

- python 3.8.10
- keras==2.8.0
- numpy==1.17.4
- opencv_python_headless==4.6.0.66
- opencv-python == 4.6.0.66
- opencv-contrib-python == 4.6.0.66
- scikit_learn==1.1.2
- tensorflow==2.8.0
- tensorflow_addons==0.16.1

## Initialization (optional)

#### Create project structure

```sh
cd src
python3 init_project.py
```

#### Generate ground truth images

```sh
cd src
python3 generate_ground_truths.py
```

Generate ground truth images that are used to compute the metrics during a model's evaluation :

- 5px and 28px dilations of the center line (`data/SECTOR/SECTOR_roads-1px.png`)

- skeleton and 20px dilation of the skeleton of the processed AMREL output (`data/SECTOR/SECTOR_amrel-processed.png`)
ages nommées `data/SECTEUR/SECTEUR_amrel-processed.png`)

## Usage

#### Train
```sh
  python3 train.py
```
_See train_parameters.py to choose parameters. In particular, proposed values of parameter_ TEST_MASK_TYPES _are provided to compare with different CNN architectures (table 1 of reference paper) or with AMREL software (table 2)._


#### Test
```sh
  python3 test_model.py
```
_See test_parameters.py to choose parameters_

#### Test with a spectific model
```sh
  python3 test_model.py [model_name]
```
model_name can be :
- unet/unet-amrel-processed-grismouton : for unet on grismouton sector
- ddcm/ddcm-amrel-processed-grismouton : for ddcm on grismouton sector
- ddcm/ddcm-cbam-dc-grismouton : for ddcm+8cbam on grismouton sector
- dcm/ddcm-cbam-grismouton : for ddcm+3cbam on grismouton sector
- ddcm/ddcm-roads-5px-grismouton : for ddcm trained with centerline dataset on grismouton sector


#### Visualize metrics during training
```sh
  tensorboard --logdir ../models
```

## Project description


### ./data

Directory containing training data
	
**Constraints**

The images must be :

- .png format
- placed in a sub-directory named according to the relevant sector
- named following the template `nameOfSector_imageType.png`

_example_

center line image for gris-mouton sector : `data/grismouton/grismouton_roads-1px.png`


**Provided images**

	slope 						slope shading of the DTM

	city-mask 				binary mask hiding urban areas

	roads-1px 				center line ground truth, manually annotated
	roads-5px					5 pixels dilation of the centerline
	roads-28px				28 pixels dilation of the centerline

	amrel-processed			AMREL output processed manually (deburring)


### ./models 

Directory containing the training results (model and logs)

**Constraints**

Subdivided according to the architecture and model names

_example_ : `./models/unet/my_model/...)`

**Files**

A training produces the following files :

- models/MODEL/MODEL_NAME/train_parameters.py : parameters used during training
- models/MODEL/MODEL_NAME/MODEL_NAME.hdf5 : trained model, can be loaded with the test script to make predictions and compute metrics
- models/MODEL/MODEL_NAME/logs/ : training logs, can be visualized with tensorboard



### ./predictions

Directory containing the images predicted during a test

**Constraints**

Subdivided according to the architecture and model names

_example_ : `./predictions/unet/my_model/...)`

**Files**

According to the chosen parameters, the following images can be produced : 

- ground_truth.png : ground truth we want to predict
- prediction_raw.png : raw output of the network 
- prediction_mask.png : network's output with a thresholding of 0.5
- prediction_overlay : green = good prediction, black = recall gt (false negative), red = false positive 

### ./src : 

Directory containing sources

**Scripts** 

- init_project.py : generates project structure (optional)
- generate_ground_truths.py : generates ground truths used to compute metrics (optional)
- train_parameters.py : training parameters (see file for more details) 
- test_parameters.py : test parameters (see file for more details) 
- train.py : train the network according to the parameters given in train_parameters.py
- test_model.py : evaluate a trained model according to the parameters given in test_parameters.py 



