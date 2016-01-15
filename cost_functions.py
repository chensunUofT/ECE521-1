from numpy import *

# Compute the current cost of the model using mean squared error
def mse_cost_function(theta, x, y):
    m = y.size
    y_model = x.dot(theta).flatten()
    error = power((y.flatten() - y_model), 2)
    J = (1.0/m) * error.sum();
    return J

# Compute the gradient of the mean square error
def mse_gradient(theta, x, y):
    m = y.size
    y_model = x.dot(theta).flatten()
    error = (y.flatten() - y_model)
    J = (1.0/m) * error.dot(x)
    return J