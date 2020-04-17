#!/bin/bash

# create env from file
conda env create -f environment.yml

# active env
conda activate lung-cancer-detection

# add env to jupyter notebook
python -m ipykernel install --user --name=lung-cancer-detection

