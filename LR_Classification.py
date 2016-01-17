from gradient_descent import *
from cost_functions import euclidean_cost_function, euclidean_gradient
from matplotlib.pyplot import *
import time

# Load Tiny MNIST dataset
with load("TINY_MNIST.npz") as data:
    x, t = data["x"], data["t"]
    x_eval, t_eval = data["x_eval"], data["t_eval"]

# Add column of biases
x = append(x,ones([shape(x)[0],1]),1)
x_eval = append(x_eval,ones([shape(x_eval)[0],1]),1)

# Linear regression parameters
iterations = 10000
lRate = 0.001
theta = zeros(x.shape[1])
errorHist = zeros(iterations)
bsize = int(raw_input("Enter mini-batch size [1]: ") or 1)
n_min, n_max = raw_input("Enter range of points to use in trainment 'min max': ").split()
n_min, n_max = int(n_min), int(n_max)
slmbda = raw_input("Enter Lambda Value [0] or 'S' to sweep values: ")
if slmbda.lower() == 's':
    print("Sweeping lambda")
    lmbda_list = [0.0001, 0.001, 0.01, 0.1, 0.5]
else:
    lmbda_list = [double(slmbda or 0)]

for lmbda in lmbda_list:
    # Define Lambda
    def derr_func(theta, x, y):
        return euclidean_gradient(theta, x, y, lmbda)

    validation_errors = zeros((n_max-n_min)/50+1)
    for k, points in enumerate(range(n_min,n_max+1,50)):
        x = x[:points]
        t = t[:points]

        print("Using %d points" %(points))
        # Perform Stochastic Gradient Descent
        start = time.time()
        epoch_error = zeros(iterations/100)
        epoch = 0
        for i in range(iterations):
            theta = stochastic_gradient_descent_step(x, t, theta, lRate, derr_func, bsize)
            errorHist[i] = euclidean_cost_function(theta, x, t, lmbda)
            if i % 100 == 0:
                for i, xi in enumerate(x_eval):
                    if round(xi.dot(theta)) != t_eval[i]:
                        epoch_error[epoch]+=1
                epoch+=1

        end = time.time()

        print("Time elapsed: %f" %(end-start))
        # Display Results

        # # Plot error history
        # subplot(1,2,1)
        # ax = plot(range(iterations), errorHist)
        # title("Cost History")
        # xlabel("Iteration")
        # ylabel("Cost")
        # grid()
        #
        # subplot(1,2,2)
        # ax = plot(range(epoch), epoch_error)
        # title("Validation Error History")
        # xlabel("Epoch")
        # ylabel("Validation Errors")
        # grid()
        #
        # show()

        # Validate model
        for i, xi in enumerate(x_eval):
            if round(xi.dot(theta)) != t_eval[i]:
                validation_errors[k]+=1
        print("Validation errors: %d" %(validation_errors[k]))
        print("\n")

    plot(range(n_min,n_max+1,50), validation_errors, 'xb-')
    xlabel("Training Points")
    ylabel("Vaidation Errors")
    grid()
    show()