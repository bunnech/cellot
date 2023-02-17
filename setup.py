# !/usr/bin/python3

from pathlib import Path
from setuptools import setup, find_packages

setup(name='cellot',
      version='0.1',
      description='Learning Single-Cell Perturbation Responses using Neural Optimal Transport',
      url='https://github.com/bunnech/cellot',
      author='Charlotte Bunne, Stefan G. Stark',
      author_email='bunnec@ethz.ch, starks@ethz.ch',
      license='BSD',
      packages=find_packages(include=['cellot', 'cellot.*']),
#       install_requires=[l.strip() for l in Path("requirements.txt").read_text("utf-8").splitlines()],
      long_description=open('README.md').read(),
      zip_safe=False,
)
