from gradient_descent import *
from cost_functions import mse_cost_function, mse_gradient
from matplotlib.pyplot import *
import time

# Generate artificial dataset
train_x = linspace(1.0, 10.0, num=100)[:, newaxis]
train_y = sin(train_x) + 0.1 * power(train_x, 2) + 0.5 * random.randn(100, 1)
train_x = train_x/max(train_x)
train_size = train_x.size

# Precalculate input powers
order = int(raw_input("Enter polynomial order: "))+1
sgd = raw_input("Use stochastic gradient descent? [y/N]").lower() == 'y'
xp = zeros(shape=(train_size, order))
for i in range(order):
    xp[:,i] = power(train_x, i)[:,0]
xpn = xp / xp.max(axis=0)


# Linear regression parameters
iterations = 40000
lRate = 0.01
theta = zeros(order)
errorHist = zeros(iterations)

if not sgd:
    # Perform Gradient Descent
    start = time.time()
    for i in range(iterations):
        theta = gradient_descent_step(xp, train_y, theta, lRate, mse_gradient)
        errorHist[i] = mse_cost_function(theta, xp, train_y)
    end = time.time()
else:
    bsize = int(raw_input("Using SGD. Enter batch size: "))
    # Perform Stochastic Gradient Descent
    start = time.time()
    for i in range(iterations):
        theta = stochastic_gradient_descent_step(xp, train_y, theta, lRate, mse_gradient, bsize)
        errorHist[i] = mse_cost_function(theta, xp, train_y)
    end = time.time()

print("Time elapsed: %f" %(end-start))
print("Final cost: %f" %(errorHist[-1]))
# Display Results
print(theta)

# Plot Adjusted Model
figure()
subplot(1,2,1)
plot_model(theta, train_x, train_y)

# Plot error history
if order == 2:
    subplot(2,2,2)
else:
    subplot(1,2,2)
ax = plot(range(iterations), errorHist)
title("Cost History")
xlabel("Iteration")
ylabel("Cost")
xscale('log')
grid()

if order == 2:
    # Plot contour of MSE
    theta0s = linspace(-10, 10, 200)
    theta1s = linspace(-3, 3, 200)
    J_vals = zeros(shape=(theta0s.size, theta1s.size))

    for t1, element in enumerate(theta0s):
        for t2, element2 in enumerate(theta1s):
            thetaT = [0, 0]
            thetaT[0] = element
            thetaT[1] = element2
            J_vals[t1, t2] = mse_cost_function(thetaT, xp, train_y)

    subplot(2,2,4)
    contour(theta0s, theta1s, J_vals, logspace(-5, 2, 10))
    scatter(theta[0], theta[1])

show()