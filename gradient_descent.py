from numpy import *
from matplotlib.pyplot import *

# Do a single step of stochastic gradient descent
def stochastic_gradient_descent_step(x, y, theta, alpha, derror, batch_size):
    random_indexes = random.permutation(y.size)[0:batch_size]
    x = x[random_indexes]
    y = y[random_indexes]
    order = shape(x)[1]
    grad = derror(theta, x, y)
    for i in range(0, order):
        theta[i] += alpha * grad[i]
    return theta

# Do a single step of gradient descent
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

# Draw contour plot of cost function with solution found
def plot_cost_function(theta, xp, train_y, costfun):
    theta1 = linspace(-theta[1]*3, theta[1]*3, 200)
    theta0 = linspace(-theta[0]*3, theta[0]*3, 200)
    J_vals = zeros(shape=(theta1.size, theta0.size))

    for t1, element in enumerate(theta1):
        for t2, element2 in enumerate(theta0):
            thetaT = [0, 0]
            thetaT[1] = element
            thetaT[0] = element2
            J_vals[t1, t2] = costfun(thetaT, xp, train_y)

    scatter(theta[1], theta[0], marker='*', color='r', s=40, label='Solution Found')
    contour(theta1, theta0, J_vals, logspace(-5,5,25), label='Cost Function')
    title("Contour Plot of Cost Function")
    xlabel(r"$\theta_1$")
    ylabel(r"$\theta_0$")
    legend(loc=2)