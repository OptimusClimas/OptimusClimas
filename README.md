# Optimus Climas
This repository contains the code for running the climate model Optimus Climas using Deep Learning techniques. We aim to optimise climate simulations with deep learning regarding regionalisation and consideration of tipping points such as the collapse of boreal permafrost and the West Antarctic Ice Sheet. We employ state-of-the-art deep learning methods, including modified vision transformers and modified gramian angular fields. Here we provide documentation on installation, setup, and a quickstart guide to reproduce our experiments and to use our GUI.
## Getting started
To set up the needed environment for Optimus Climas, we use ```python>=3.11```. To create the environment use requirements.txt.
All our models and datasets needed for the GUI are hosted on [Hugging Face](https://huggingface.co/collections/OptimusClimas/files-for-gui-678192abaf887f31d684639e). First create an empty folder named ```models``` in ```KlimaUi/climatesimulationAI``` and an empty folder named ```trainingdata``` in ```KlimaUi/climatesimulationAI/Training/PreProcessing```.Please download the [models](https://huggingface.co/OptimusClimas/models) and put them in ```KlimaUi/climatesimulationAI/models```. Please download the dataset [trainingdata](https://huggingface.co/datasets/OptimusClimas/trainingdata) and put them in ```KlimaUi/climatesimulationAI/Training/PreProcessing/trainingdata```. Please set the working directory to ```YOURLOCALPATH\KlimaUi```.

To use the Graphical User Interface of Optimus Climas start ```KlimaUi/ui_OptimusKlimasEntry.py. 
To reproduce the figures in our publication see figures.ipynb.
To run your own experiments see manualuse.ipynb.
