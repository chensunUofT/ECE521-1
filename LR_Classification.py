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
iterations = 5000
lRate = 0.01
bsize = int(raw_input("Enter mini-batch size [50]: ") or 50)
n_min, n_max = (raw_input("Enter range of points to use in trainment 'min max' [50 800]: ") or "50 800").split()
n_min, n_max = int(n_min), int(n_max)
verbose_plot = raw_input("Plot cost and validation for each iteration [y/N]? ").lower() == 'y'
slmbda = (raw_input("Enter Lambda value or 'S' to sweep values [s]: ") or "s")
if slmbda.lower() == 's':
    print("Sweeping lambda")
    lmbda_list = [0, 0.0001, 0.001, 0.01, 0.1]
else:
    lmbda_list = [double(slmbda)]

for lmbda in lmbda_list:
    # Define Lambda
    def derr_func(theta, x, y):
        return euclidean_gradient(theta, x, y, lmbda)

    validation_errors = zeros((n_max-n_min)/50+1)
    for k, points in enumerate(range(n_min,n_max+1,50)):
        theta = zeros(x.shape[1])
        xk = x[:points]
        tk = t[:points]

        # Perform Stochastic Gradient Descent
        epoch_iterations = (points / bsize)
        epochs = iterations / epoch_iterations
        epoch_error = zeros(epochs)
        errorHist = zeros(iterations)
        start = time.time()
        for j in range(epochs):
            for i in range(epoch_iterations):
                xp = xk[i*bsize:(i+1)*bsize]
                tp = tk[i*bsize:(i+1)*bsize]
                theta = gradient_descent_step(xp, tp, theta, lRate, derr_func)
                errorHist[(j*epoch_iterations)+i] = euclidean_cost_function(theta, x, t, lmbda)
            for i, xi in enumerate(x_eval):
                if round(xi.dot(theta)) != t_eval[i]:
                    epoch_error[j] += 1
        end = time.time()
        if(verbose_plot):
            # Plot error history
            subplot(1,2,1)
            ax = plot(range(iterations), errorHist, '-')
            title("Cost History N=%d" %(points))
            xlabel("Iteration")
            ylabel("Cost")
            grid()
            # Plot validation error history
            subplot(1,2,2)
            ax = plot(range(epochs), epoch_error)
            title("Error History N=%d" %(points))
            xlabel("Epoch")
            ylabel("Validation Errors")
            grid()
            show()


        # Validate model
        for i, xi in enumerate(x_eval):
            if round(xi.dot(theta)) != t_eval[i]:
                validation_errors[k]+=1
        print("Lambda = %f, N = %d, Validation errors: %d, Time = %f" %(lmbda, points, validation_errors[k], end-start))

    print("\n")
    plot(range(n_min,n_max+1,50), validation_errors, 'x-', label=r"$\lambda$=%f"%(lmbda))
    xlabel("Training Points")
    ylabel("Vaidation Errors")

grid()
legend(loc='best')
show()
