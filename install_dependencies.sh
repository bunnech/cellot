#!/usr/bin/python3

conda create --name cell python=3.9.5
conda activate cell

conda update -n base -c defaults conda
pip install --upgrade pip

pip install -r requirements.txt
python setup.py develop
