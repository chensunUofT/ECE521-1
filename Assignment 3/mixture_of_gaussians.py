import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from time import time
from math import pi
from utils import reduce_logsumexp, logsoftmax
from distance_functions import square_distance

def log_density(x, mu, sigma):
    '''
        Calculates log P(x | mu, sigma), i.e. the log probability density
        function for the mixture of gaussians
    Args:
        x: Data points
        mu: Cluster centers
        sigma: Cluster variance

    Returns:
        Log probability density function of the data
    '''
    den = tf.sqrt(2 * pi * sigma) # sqrt(2pi*sigma^2)
    D = tf.to_float(tf.rank(x))
    logp = -D*tf.log(den) # 1/(sqrt(2pi) * sigma)
    dists = square_distance(x, mu) # (x - mu)^2
    logp -= dists / (2*sigma) # 1/den - (x - mu)^2 / (2*sigma^2)

    return logp


def log_cluster_probability(x, logpz, mu, sigma):
    '''
        Computes the vector of log posterior cluster probabilities given the
        data, prior, mean and variance. P(Z | x)
    Args:
        x: Data
        pz: Prior probabilities
        mu: Gaussian mean
        sigma: Gaussian variance

    Returns:
        Vector of log posterior cluster probabilities
    '''

    logpxgz = log_density(x, mu, sigma) # logP(x | z)
    num = logpz + tf.transpose(logpxgz)
    den = reduce_logsumexp(num, 0)
    logpzgx = num/den # pz * P(x | z) / P(x)

    return logpzgx


def marginal_log_likelihood(x, logpz, mu, sigma):
    '''
        Calculates the log marginal probability of the data, i.e. logP(x)
    Args:
        x: The data
        pz: Prior probabilities of the clusters
        mu: Cluster means
        sigma: Cluster variance

    Returns:
        Log marginal probability of the data
    '''
    pxn = reduce_logsumexp(tf.transpose(logpz) + log_density(x, mu, sigma),0)
    px = tf.reduce_sum(pxn,0)

    return px


def train_mog(data, k, EXP=1e-5):
    '''
        Trains a single mixture of gaussian model for a zero-mean,
        unit variance normalized data.
    Args:
        x: Data (numpy array)
        k: Number of clusters

    Returns:
        Optimal cluster parameters
    '''
    data_len = len(data)
    assert (data_len > 0), "Dataset is empty"
    data_d = len(data[0])
    tf.set_random_seed(time())

    pz = tf.Variable(tf.zeros([1,k]))
    logpz = logsoftmax(pz) # Enforce simplex constraint over P(z)

    sigma = tf.Variable(tf.ones([k, 1])*(-3))
    expsigma = tf.exp(sigma) # Enforce sigma > 0

    mu = tf.Variable(tf.random_normal([k, data_d], mean=0, stddev=0.01))
    x = tf.placeholder(tf.float32, [None, len(data[0])])

    cost = -marginal_log_likelihood(x, logpz, mu, expsigma)

    iter_var = tf.Variable(0)
    optimizer = tf.train.AdamOptimizer(0.03, beta1=0.9, beta2=0.99, epsilon=1e-5)
    train = optimizer.minimize(cost, global_step=iter_var)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    with sess.as_default():
        costs = []
        best_cost = float('inf')
        last_cost = float('inf')
        print "------------------"
        print "P(x):  ", tf.exp(logpz).eval()
        print "Sigma: ", expsigma.eval().reshape((1,k))
        print "Mu:    ", mu.eval()
        print "------------------"
        try:
            while True:
                iter_cost = sess.run([cost, train], feed_dict={x: data})[0]

                iter = iter_var.eval()
                costs.append(iter_cost)
                if iter % 100 == 0:
                    print "Iteration:", iter
                    print "Log Likelihood:", -iter_cost

                if iter_cost < best_cost:
                    best_cost = iter_cost
                    clusters = [logpz.eval(), mu.eval(), expsigma.eval()]

                if iter > 5000 or abs(iter_cost - last_cost) < EXP:
                    print "Converged!"
                    break
                else:
                    last_cost = iter_cost

        except KeyboardInterrupt:
            if len(clusters) == 0:
                clusters = [logpz.eval(), mu.eval(), expsigma.eval()]

    return clusters, costs


def train_mog_model(data, k, validation, numtries=5):
    sess = tf.Session()
    valcost = -float('inf')
    fcosts = []
    for i in range(numtries):
        (logpz, mu, sigma), costs = train_mog(data, k)
        with sess.as_default():
            valcost_i = marginal_log_likelihood(validation, logpz, mu, sigma).eval()
            print "Validation Log Likelihood: %f" %valcost_i
        if valcost_i > valcost:
            print "New Best!"
            fclusters = [logpz, mu, sigma]
            valcost = valcost_i
            fcosts = costs

    print "Best cost: %f" %valcost

    with sess.as_default():
        logpx = log_cluster_probability(validation, *fclusters)
        px = tf.nn.softmax(logpx)
        assignments = px.eval()

    return fclusters, assignments, fcosts, valcost



def plot_mog(dataset, means, sigma, classes, ax1=None):
    '''

    Draws a scatterplot of the clusterized data
    Args:
        dataset: Dataset to be plotted
        clusters: Clusters centers
        classes: Clases of each point in the dataset

    '''
    k = len(means)
    cs = range(k)
    classes = np.argmax(classes,1)
    print classes
    x = np.arange(-3, 4, 0.025)
    y = np.arange(-4, 2.5, 0.025)
    X, Y = np.meshgrid(x, y)
    cm = plt.get_cmap('Set1', k)
    if ax1 is None:
        ax1 = plt.gca()
    ax1.scatter(dataset[:, 0], dataset[:, 1], c=classes, s=25, alpha=0.6, cmap=cm, label="Data")
    ax1.scatter(means[:, 0], means[:, 1], marker='*', c=cs, s=250, linewidths=3, cmap=cm, label="Clusters")
    for i in range(len(sigma)):
        Z = mlab.bivariate_normal(X, Y, sigma[i], sigma[i], means[i,0], means[i,1])
        ax1.contour(X,Y,Z, colors=[cm(i)])
    plt.title('Mixture of Gaussians')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid()
