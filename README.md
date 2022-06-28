# [Learning Single-Cell Perturbation Responses using Neural Optimal Transport](https://www.biorxiv.org/content/10.1101/2021.12.15.472775v1.full.pdf)

**Authors**: Charlotte Bunne\*, Stefan G. Stark\*, Gabriele Gut\*, Jacobo Sarabia Del Castillo, Kjong-Van Lehmann, Lucas Pelkmans, Andreas Krause, Gunnar Rätsch

The preprint is available [**here**](https://www.biorxiv.org/content/10.1101/2021.12.15.472775v1.full.pdf).

<p align='center'><img src='assets/overview.png' alt='Overview.' width='100%'> </p>

Understanding and predicting molecular responses in single cells upon chemical, genetic, or mechanical perturbations is a core question in biology. Obtaining single-cell measurements typically requires the cells to be destroyed. This makes learning heterogeneous perturbation responses challenging as we only observe *unpaired* distributions of perturbed or non-perturbed cells. Here we leverage the theory of optimal transport and the recent advent of convex neural architectures to present `CellOT`, a framework for learning the response of individual  cells to a given perturbation by coupling these unpaired distributions. We achieve this alignment with a learned transport map that allows us to infer the treatment responses of unseen untreated cells. `CellOT` outperforms current methods at predicting single-cell drug responses, as profiled by scRNA-seq and a multiplexed protein imaging technology.

This repository contains the `CellOT` method and evaluation scripts to reproduce the results of experiments on predicting single-cell drug responses, as profiled by scRNA-seq and a multiplexed protein imaging technology. Further, we provide experiments on `CellOT` generalization performance to unseen settings by (a) predicting the scRNA-seq responses of holdout lupus patients exposed to IFN-beta, and (b) modeling the hematopoietic developmental trajectories of different subpopulations.

## Installation

To install `CellOT` run:
```
python setup.py develop
```
Package requirements and dependencies are listed in `requirements.txt`.

## Datasets
You can download the preprocessed data [here]().

## Citation

In case you found our work useful, please consider citing us:
```
@article{bunne2021learning,
  title={{Learning Single-Cell Perturbation Responses using Neural Optimal Transport}},
  author={Bunne, Charlotte and Stark, Stefan G and Gut, Gabriele and del Castillo, Jacobo Sarabia and Lehmann, Kjong-Van and Pelkmans, Lucas and Krause, Andreas and Ratsch, Gunnar},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Contact
In case you have questions, please contact [Stefan G. Stark](mailto:starks@ethz.ch) and [Charlotte Bunne](mailto:bunnec@ethz.ch).
