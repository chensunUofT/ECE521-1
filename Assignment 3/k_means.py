import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import numpy as np
from distance_functions import square_distance


def get_cost(data, clusters):
    '''

    Calculates the loss for a given data and set of clusters

    Args:
        data: Data set
        clusters: Cluster centers

    Returns:
        Loss function value

    '''
    data = tf.Variable(data.astype('float32'))
    clusters = tf.Variable(clusters.astype('float32'))
    cost = loss_function(clusters, data)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    return sess.run(cost)


def plot_data(dataset, clusters, classes):
    '''

    Draws a scatterplot of the clusterized data
    Args:
        dataset: Dataset to be plotted
        clusters: Clusters centers
        classes: Clases of each point in the dataset

    '''
    k = len(clusters)
    plt.scatter(dataset[:, 0], dataset[:, 1], c=classes, cmap=plt.get_cmap('Set1'), s=25, alpha=0.6)
    plt.scatter(clusters[:, 0], clusters[:, 1], marker='*', c=range(k), cmap=plt.get_cmap('Set1'), s=500, linewidths=3)
    plt.title('K-Means Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid()


def loss_function(mu, data):
    '''

    Definition of the loss function for K-Means Clustering
        L(mu) = sum_n( min_k( x_n - mu_k ) )

    Args:
        mu: Data cluster centers
        data: Data set

    Returns:
        cost: Definition of the loss

    '''
    dists = square_distance(mu, data)  # ||x - mu||
    min_dist = tf.reduce_min(dists, 1)  # min ||x - mu||
    cost = tf.reduce_mean(min_dist)  # sum(min ||x - mu||)
    return cost


def assign_data(dataset, clusters):
    '''

    Calculates the cluster assignments given a dataset and cluster centers
    Args:
        dataset: Set of points
        clusters: Centers of clusters

    Returns:
        min_dist: List of point classes in the same order they appear in the dataset

    '''
    dists = square_distance(clusters, dataset)  # ||x - mu||
    min_dist = tf.argmin(dists, 1)  # argmin ||x - mu||
    return min_dist


def k_means(data, k, EXP=1e-6):
    '''

    Performs K-Means clusterization on a dataset
    Args:
        data: Set of points
        k: Number of clusters to use
        EXP: Convergence criteria (minimum difference between iteration cost before stopping

    Returns:
        clusters: Cluster centers
        assignments: Class assignment of each point
        costs: Cost history throughout training

    '''
    data_len = len(data)
    assert (data_len > 0), "Dataset is empty"
    assert (k < data_len), "Invalid value of K for size of dataset"

    dim = len(data[0])

    # Input data and Clusters
    dataset = tf.placeholder(tf.float32, [None, dim], name="Data")
    mu = tf.Variable(tf.random_normal([k, dim]), name="Centroids")

    # Training specification
    cost = loss_function(mu, dataset)
    iter_var = tf.Variable(0)
    optimizer = tf.train.AdamOptimizer(0.1, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(cost, global_step=iter_var)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    with sess.as_default():
        costs = []
        last_cost = float('inf')
        while True:
            iter_cost = sess.run([cost, optimizer], feed_dict={dataset: data})[0]

            iter = iter_var.eval()
            costs.append(iter_cost)
            if iter % 50 == 0:
                print "Iteration:", iter
                print "Cost:", iter_cost, "\n"

            if abs(iter_cost - last_cost) < EXP:
                print "Converged!"
                clusters = mu.eval()
                break
            else:
                last_cost = iter_cost

        assignments = sess.run(assign_data(dataset, mu), feed_dict={dataset: data})

    return clusters, assignments, costs


def evaluate_cost_function():
    '''

    Quick and dirty implementation of an experiment to draw the shape of the cost function.
    Generates a random, normal distributed set of 1000 1D points and computes the cost for
    100*100 possible pairs of clusters from -5 to 5. Performance is poor.

    Returns:
        values: List with clusters and costs [cluster0, cluster1, cost]

    '''
    nsamples = 100
    theta1 = np.linspace(-5, 5, nsamples)
    theta0 = np.linspace(-5, 5, nsamples)
    data = tf.Variable(tf.random_normal([1000, 1]))
    thetaT = [0, 0]
    thetaT[1] = theta1[0]
    thetaT[0] = theta0[1]
    thetaT = np.array([thetaT]).T.astype('float32')
    clusters = tf.Variable(thetaT)
    cost = loss_function(clusters, data)
    total = len(theta1) * len(theta0)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    values = np.zeros((nsamples ** 2, 3))
    with sess.as_default():
        for t1, element in enumerate(theta1):
            for t2, element2 in enumerate(theta0):
                thetaT = [0, 0]
                thetaT[1] = element
                thetaT[0] = element2
                thetaT = np.array([thetaT]).T.astype('float32')
                sess.run(clusters.assign(thetaT))
                cost_i, cluster = sess.run([cost, clusters])
                iter = (t1 * len(theta1) + t2)
                values[iter] = np.array((cluster[0], cluster[1], cost_i))

    return values
