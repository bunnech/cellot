# !/usr/bin/python3

from setuptools import setup

setup(name='cellot',
      version='0.1',
      description='Learning Single-Cell Perturbation Responses using Neural Optimal Transport',
      url='https://github.com/bunnech/cellot',
      author='Charlotte Bunne, Stefan G. Stark',
      author_email='bunnec@ethz.ch, starks@ethz.ch',
      license='BSD',
      install_requires=[],
      py_modules=['cellot'],
      long_description=open('READme.md').read(),
      zip_safe=False,
)
