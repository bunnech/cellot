#!/usr/bin/python3

conda create --name cellot python=3.9.5
conda activate cellot

conda update -n base -c defaults conda
pip install --upgrade pip

pip install -r requirements.txt
python setup.py develop
