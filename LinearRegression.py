from gradient_descent import *
from cost_functions import mse_cost_function, mse_gradient
from matplotlib.pyplot import *

# Generate artificial dataset
train_x = linspace(1.0, 10.0, num=100)[:, newaxis]
train_y = sin(train_x) + 0.1 * power(train_x, 2) + 0.5 * random.randn(100, 1)
train_x = train_x/max(train_x)
train_size = train_x.size

# Precalculate input powers
order = int(raw_input("Enter polynomial order [1]: ") or 1)+1
sgd = raw_input("Use stochastic gradient descent? [y/N] ").lower() == 'y'
xp = zeros(shape=(train_size, order))
for i in range(order):
    xp[:,i] = power(train_x, i)[:,0]
xpn = xp / xp.max(axis=0)


# Linear regression parameters
iterations = 10000
conv = 1e-6
theta = zeros(order)
converged = False

if not sgd:
    # Perform Gradient Descent
    lRate = 0.5
    errorHist = []
    i = 0
    while i < iterations and not converged:
        ltheta = theta
        theta = gradient_descent_step(xp, train_y, theta, lRate, mse_gradient)
        errorHist.append(mse_cost_function(theta, xp, train_y))
        if i>1:
            if errorHist[-2] - errorHist[-1] < conv:
                converged = True
                print "Converged after %d iterations" %(i)
        i+=1
else:
    bsize = int(raw_input("Using SGD. Enter batch size [10]: ") or 10)
    # Perform Stochastic Gradient Descent
    iterations /= bsize
    lRate = 0.1
    errorHist = []
    i = 0
    while i < iterations and not converged:
        ltheta = theta
        theta = stochastic_gradient_descent_epoch(xp, train_y, theta, lRate, mse_gradient, bsize)
        errorHist.append(mse_cost_function(theta, xp, train_y))
        if i>2:
            if errorHist[-2] - errorHist[-1] < conv:
                converged = True
                print "Converged after %d epochs" %(i)
        i+=1

print("Final cost: %f" %(errorHist[-1]))
# Display Results
print(theta)

# Plot Adjusted Model
subplot(1,2,1)
plot_model(theta, train_x, train_y)

# Plot error history
if order == 2:
    ax = subplot(2,2,2)
else:
    ax = subplot(1,2,2)

plot(range(i), errorHist)
title("Cost History")
xlabel("Iteration")
ylabel("Cost")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.97, 0.95, "Final cost: %f" %(errorHist[-1]), transform=ax.transAxes, fontsize=14,
        verticalalignment='top', ha='right', bbox=props)
grid()

if order == 2:
    # Plot contour of MSE
    subplot(2,2,4)
    plot_cost_function(theta, xp, train_y, mse_cost_function)
    tight_layout()
    savefig("results/Task_3.eps")
    show()
    plot_equation(theta, "results/Task_3_eq.pgf")

else:
    tight_layout()
    savefig("results/Task_4.eps")
    show()
    plot_equation(theta, "results/Task_4_eq.pgf")
