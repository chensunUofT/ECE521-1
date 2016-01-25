from numpy import *
from matplotlib.pyplot import *

# Do an epoch of stochastic gradient descent
def stochastic_gradient_descent_epoch(x, y, theta, alpha, derror, batch_size):
    batch = y.size
    epochs = batch / batch_size
    random_indexes = random.permutation(batch)
    x = x[random_indexes]
    y = y[random_indexes]
    order = shape(x)[1]
    for k in range(epochs):
        xk = x[k*batch_size:(k+1)*batch_size]
        yk = y[k*batch_size:(k+1)*batch_size]
        grad = derror(theta, xk, yk)
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

def plot_equation(theta, file):
    '''
    Adapted from http://stackoverflow.com/questions/14110709/creating-images-of-mathematical-expressions-from-tex-using-matplotlib
    '''
    formula = "$f(x) = %f" %(theta[0])
    for i, coef in enumerate(theta[1:]):
        formula += " + (%f) \cdot x^%d" %(coef, i+1)
    formula += "$"

    fig = figure()
    text = fig.text(0, 0, formula)
    dpi = 300
    fig.savefig(file, dpi=dpi)

    bbox = text.get_window_extent()
    width, height = bbox.size / float(dpi) + 0.005
    fig.set_size_inches((width, height))

    dy = (bbox.ymin/float(dpi))/height
    text.set_position((0, -dy))

    fig.savefig(file, dpi=dpi)