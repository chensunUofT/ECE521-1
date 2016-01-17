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
    n_list = [double(sn)]

sk = raw_input("Enter number of nearest neighbors to consider or 'S' to sweep values [S]: ") or 's'
if sk.lower() == 's':
    print("Sweeping k")
    k_list = [1, 3, 5, 7, 21, 101, 401]
else:
    k_list = [double(sk)]

for n in n_list:
    validation_errors = zeros(size(k_list))
    for j, k in enumerate(k_list):
        xj = x[:n]
        tj = t[:n]
        for i, xi in enumerate(x_eval):
            ti = knn_classify(xj, tj, xi, k, euclidean_distance)
            if ti != t_eval[i]:
                validation_errors[j] += 1
        print("K = %d, N = %d, Validation Errors = %d"%(k,n,validation_errors[j]))
    print(" ")
    plot(k_list, validation_errors, 'xb-')
    title('N=%d'%(n))
    xlabel('Neighbors Considered')
    ylabel('Validation Errors')
    show()