from gradient_descent import *
from cost_functions import euclidean_cost_function, euclidean_gradient
from matplotlib.pyplot import *

# Load Tiny MNIST dataset
with load("TINY_MNIST.npz") as data:
    x, t = data["x"], data["t"]
    x_eval, t_eval = data["x_eval"], data["t_eval"]

# Add column of biases
x = append(x,ones([shape(x)[0],1]),1)
x_eval = append(x_eval,ones([shape(x_eval)[0],1]),1)

# Linear regression parameters
iterations = 10000
lRate = 0.01
bsize = int(raw_input("Enter mini-batch size [50]: ") or 50)

sn = (raw_input("Enter train set size or 'S' to sweep values [s]: ") or "s")
if sn.lower() == 's':
    print("Sweeping N")
    nlist = [50, 100, 200, 400, 800]
else:
    nlist = [double(sn)]

verbose_plot = raw_input("Plot classification error history for each iteration [y/N]? ").lower() == 'y'

slmbda = (raw_input("Enter Lambda value or 'S' to sweep values [s]: ") or "s")
if slmbda.lower() == 's':
    print("Sweeping lambda")
    lmbda_list = [0, 0.0001, 0.001, 0.01, 0.1, 0.5]
else:
    lmbda_list = [double(slmbda)]

validation_errors = zeros((len(nlist), len(lmbda_list)))
for l, lmbda in enumerate(lmbda_list):
    # Define Lambda
    def derr_func(theta, x, y):
        return euclidean_gradient(theta, x, y, lmbda)

    for k, points in enumerate(nlist):
        theta = zeros(x.shape[1])
        xk = x[:points]
        tk = t[:points]

        # Perform Stochastic Gradient Descent
        epoch_iterations = int(ceil(points / bsize))
        epochs = int(ceil(iterations / epoch_iterations))
        epoch_error_eval = []
        epoch_error_train = []
        errorHist = []
        conv = 1e-6
        converged = False
        j=0
        while j < epochs and not converged:
            for i in range(epoch_iterations):
                xp = xk[i*bsize:(i+1)*bsize]
                tp = tk[i*bsize:(i+1)*bsize]
                theta = gradient_descent_step(xp, tp, theta, lRate, derr_func)
            errorHist.append(euclidean_cost_function(theta, xp, tp, lmbda))
            if j>1:
                if abs(errorHist[-2] - errorHist[-1]) < conv:
                    converged = True
            if(verbose_plot):
                epoch_error_eval.append(0)
                epoch_error_train.append(0)
                for i, xi in enumerate(x_eval):
                    if round(xi.dot(theta)) != t_eval[i]:
                        epoch_error_eval[j] += 1
                for i, xi in enumerate(x):
                    if round(xi.dot(theta)) != t[i]:
                        epoch_error_train[j] += 1
            j+=1
        if(verbose_plot):
            # Plot classification error history
            plot(range(j), epoch_error_eval, label="Evaluation Set")
            plot(range(j), epoch_error_train, label="Train Set")
            title("Error History N=%d" %(points))
            xlabel("Epoch")
            ylabel("Classification Errors")
            legend(loc='best')
            grid()
            savefig("results/Task_6.eps")
            show()

            # Plot cost history - Debugging
            plot(range(j), errorHist)
            title("Cost History N=%d" %(points))
            xlabel("Epoch")
            ylabel("Cost")
            legend(loc='best')
            grid()
            show()


        # Validate model
        for i, xi in enumerate(x_eval):
            pred = 1 if xi.dot(theta) > 0.5 else 0
            if pred != t_eval[i]:
                validation_errors[k, l]+=1
        print("Lambda = %f, N = %d, Validation errors: %d, Epochs for Convergence: %d"
              %(lmbda, points, validation_errors[k, l], j))

if len(nlist) > 1 and len(lmbda_list) > 1:
    for i, lmbda in enumerate(lmbda_list):
        plot(nlist, validation_errors[:,i], 'x-', label=r"$\lambda$=%f"%(lmbda))
    legend(loc='best')
    xlabel("Training Points")
    ylabel("Validation Errors")
    grid()
elif len(nlist) > 1:
    plot(nlist, validation_errors[:,0], 'x-')
    xlabel("Training Points")
    ylabel("Validation Errors")
    grid()
    savefig("results/Task_5.eps")
    savetxt("results/Task_5.csv", array(zip(nlist, validation_errors[:,0])), fmt='%i %i')
elif len(lmbda_list) > 1:
    plot(lmbda_list, validation_errors[0,:], 'x-')
    xlabel(r"$\lambda$")
    ylabel("Validation Errors")
    xscale('log')
    grid(True,which="both",ls="-")
    savefig("results/Task_7.eps")
    savetxt("results/Task_7.csv", array(zip(lmbda_list, validation_errors[0,:])), fmt='%.4f %i')

show()
