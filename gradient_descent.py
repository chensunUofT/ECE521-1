from numpy import *
from matplotlib.pyplot import *

# Do a single step of stochastic gradient descent considering quadratic error
def stochastic_gradient_descent_step(x, y, theta, alpha, derror, batch_size):
    random_indexes = random.permutation(y.size)[0:batch_size]
    x = x[random_indexes]
    y = y[random_indexes]
    order = shape(x)[1]
    grad = derror(theta, x, y)
    for i in range(0, order):
        theta[i] += alpha * grad[i]
    return theta

# Do a single step of gradient descent considering quadratic error
def gradient_descent_step(x, y, theta, alpha, derror):
    order = shape(x)[1]
    grad = derror(theta, x, y)
    for i in range(0, order):
        theta[i] += alpha * grad[i]
    return theta

# Plot the fitted model against the original data
def plot_model(theta, x, y):
    scatter(x, y, marker='x', color='r', label='Training Data')
    result = polyval(theta[::-1], x)
    plot(x, result, label='Fitted Polynomial')
    title("Linear Regression")
    legend(loc=2)
    grid()