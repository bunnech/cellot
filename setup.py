# !/usr/bin/python3

from setuptools import setup, find_packages
import os

setup(name='cellot',
      version='0.1',
      description='Learning Single-Cell Perturbation Responses using Neural Optimal Transport',
      url='https://github.com/bunnech/cellot',
      author='Charlotte Bunne, Stefan G. Stark',
      author_email='bunnec@ethz.ch, starks@ethz.ch',
      packages=['cellot'],
      license='BSD',
      packages=find_packages(),
      install_requires=[],
      long_description=open('README.txt').read(),
      zip_safe=False,
)
