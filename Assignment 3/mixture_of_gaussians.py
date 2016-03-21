import tensorflow as tf
import numpy as np
from math import pi
from utils import reduce_logsumexp
from distance_functions import square_distance


def log_density(x, mu, sigma):
    '''
        Calculates log P(x | mu, sigma), i.e. the log probability density
        function for the mixture of gaussians
    Args:
        x: Data points
        mu: Cluster centers
        sigma: Cluster standard deviation

    Returns:
        Log probability density function of the data
    '''

    den = np.sqrt(2 * pi) # sqrt(2pi)

    logp = tf.log(1/(sigma * den)) # 1/(sqrt(2pi) * sigma)
    dists = square_distance(x, mu) # (x - mu)^2
    logp -= dists / (2*tf.pow(sigma, 2)) # 1/den - (x - mu)^2 / (2*sigm
    # a^2)

    return logp


def log_cluster_probability(x, pz, mu, sigma):
    '''
        Computes the vector of log posterior cluster probabilities given the
        data, prior, mean and standard deviation. P(Z | x)
    Args:
        x: Data
        pz: Prior probabilities
        mu: Gaussian mean
        sigma: Gaussian standard deviation

    Returns:
        Vector of log posterior cluster probabilities
    '''

    logpxgz = log_density(x, mu, sigma) # logP(x | z)
    logpz = tf.log(pz)
    num = logpz + logpxgz # logP(z) + logP(x | z)
    den = logpxgz + logpz
    den = reduce_logsumexp(den, reduction_indices=0)
    logpzgx = num/den # pz * P(x | z) / P(x)

    return tf.reduce_sum(logpzgx, reduction_indices=1)


def marginal_log_likelihood(x, pz, mu, sigma):
    '''
        Calculates the marginal log probability of X, i.e. logP(x).
    Returns:

    '''
    pass