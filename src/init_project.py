
""" Create project structure """
import os

paths = [
	"../data/cuveaux",
	"../data/grandrupt",
	"../data/grismouton",
	"../data/stmont",
#	"../models/ddcm",
#	"../models/unet",
	"../predictions/ddcm",
	"../predictions/unet"
]

for path in paths : 
	if not os.path.exists(path):
	    os.makedirs(path)


