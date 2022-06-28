#!/usr/bin/python3
# author: Charlotte Bunne

# imports
import numpy as np
import ot


def wasserstein_loss(x, y, epsilon=0.5):
    """Computes transport between (x, a) and (y, b) via Sinkhorn algorithm."""
    # uniform distribution on samples
    a = np.ones(len(x)) / len(x)
    b = np.ones(len(y)) / len(y)

    # compute loss matrix
    cost = ot.dist(x, y)

    # compute reg Wasserstein distance
    dist = ot.sinkhorn2(a, b, cost, epsilon, method='sinkhorn_stabilized')
    return dist
