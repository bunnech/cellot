# !/usr/bin/python3

from setuptools import setup, find_packages
import os

setup(name='cellot',
      version='0.1',
      description='Learning Single-Cell Perturbation Responses using Neural Optimal Transport',
      url='https://github.com/bunnech/cellot',
      author='Charlotte Bunne, Stefan G. Stark',
      author_email='bunnec@ethz.ch, starks@ethz.ch',
      license='BSD',
      packages=find_packages(),
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=[],
      zip_safe=False,
)
