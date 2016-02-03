from distance_functions import euclidean_distance
from gradient_descent import *
from matplotlib.pyplot import *
import time

# Load Tiny MNIST dataset
from k_nearest_neighbors import knn_classify

with load("TINY_MNIST.npz") as data:
    x, t = data["x"], data["t"]
    x_eval, t_eval = data["x_eval"], data["t_eval"]

sn = raw_input("Enter number of points to use in training or 'S' to sweep values [S]: ") or 's'
if sn.lower() == 's':
    print("Sweeping N")
    n_list = [5, 50, 100, 200, 400, 800]
else:
    n_list = [int(sn)]

sk = raw_input("Enter number of nearest neighbors to consider or 'S' to sweep values [S]: ") or 's'
if sk.lower() == 's':
    print("Sweeping k")
    k_list = [1, 3, 5, 7, 21, 101, 401]
else:
    k_list = [int(sk)]


validation_errors = zeros((len(k_list), len(n_list)))
for l, n in enumerate(n_list):
    for j, k in enumerate(k_list):
        xj = x[:n]
        tj = t[:n]
        start = time.time()
        for i, xi in enumerate(x_eval):
            ti = knn_classify(xj, tj, xi, k, euclidean_distance)
            if ti != t_eval[i]:
                validation_errors[j,l] += 1
        end = time.time()
        print("K = %d, N = %d, Validation Errors = %d, Time = %f"%(k,n,validation_errors[j,l],end-start))
    print(" ")

if len(k_list) > 1:
    plot(k_list, validation_errors[:,0].flatten()/len(x_eval)*100, 'x-', label="N = %d"%(n_list[0]))
    xlabel('Neighbors Considered')
    ylabel('Validation Errors %')
    legend(loc='best')
    xscale('log')
    grid(True,which="both",ls="-")
    savefig("results/Task_2.eps")
    savetxt("results/Task_2.csv", array(zip(k_list, validation_errors[:,0])), fmt='%i %i')
elif len(n_list) > 1:
    plot(n_list, validation_errors[0,:].flatten()/len(x_eval)*100, 'x-', label="K = %d"%(k_list[0]))
    xlabel('Train Set Size')
    ylabel('Validation Errors %')
    legend(loc='best')
    grid()
    savefig("results/Task_1.eps")
    savetxt("results/Task_1.csv", array(zip(n_list, validation_errors[0,:])), fmt='%i %i')

show()
