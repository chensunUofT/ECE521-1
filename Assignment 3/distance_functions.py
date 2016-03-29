import tensorflow as tf






def matrix_norm_column(x):
    '''
    Calculates the norm of each vector in a matrix, returning a column matrix
    '''
    x_sq = x*x
    norm = tf.reduce_sum(x_sq, 1, keep_dims=True)
    return norm

def square_distance(x, y):
    '''
    Returns pairwise squared distance between two matrices:
    For points xi belonging to X and yj belonging to Y, we have:
        d(xi,yj)^2 = ||xi - yj||^2 = ||xi||^2 + ||xj||^2 - 2 * (yj . xi)
    Expanding to the entire matrix, we get:
        d(X, Y)^2 = (||X||^2 .+ ||Y||^2) - 2*dot(Y,X')
            Where .+ is the outer (element-wise) sum
    '''

    x_norm = matrix_norm_column(x)
    y_norm = matrix_norm_column(y)
    dot_prod = tf.matmul(y, tf.transpose(x))

    outer_sum = tf.transpose(x_norm) + y_norm

    return outer_sum - 2 * dot_prod