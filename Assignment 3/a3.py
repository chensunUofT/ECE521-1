from k_means import k_means, plot_data, evaluate_cost_function, get_cost
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import matplotlib.pyplot as plt


# Create Results Dir
if not os.path.exists("results"):
    os.makedirs("results")

# Load dataset
data = np.load('data2D.npy')
np.random.shuffle(data)
points = len(data)
validation = data[:points/3]
dataset = data[points/3:]

'''
PART 1.2.1
'''
#vals2 = evaluate_cost_function()
costs = np.load('costs.npy')
fig = plt.figure()
ax = fig.add_axes([.05, .05, .9, .9], projection='3d')
ax.plot_trisurf(costs[::5,0], costs[::5,1], costs[::5,2], cmap=plt.cm.Spectral, shade=False, linewidth=0.1)
ax.set_xlabel('Cluster 1')
ax.set_ylabel('Cluster 2')
ax.set_zlabel('Cost')
plt.grid()
plt.savefig('results/cost_function.pdf', dpi=50)
plt.close()


'''
PARTS 1.2.2, 1.2.3, 1.2.4
'''
krange = range(1,6)
percentages = np.zeros((len(krange), max(krange)+1))
val_costs = np.zeros((len(krange), 2))
lclusters, lclasses = [], []
for k in krange:
    clusters, classes, costs = k_means(dataset, k)

    percentages[k-1][0] = k
    for kclass in range(k):
        percentages[k-1][kclass+1] = np.sum(np.equal(classes, kclass)) * 100.0/len(classes)

    val_costs[k-1] = np.array([k, get_cost(validation, clusters)])

    # Make separate plot of clusters when K = 3
    if k == 3:
        plot_data(dataset, clusters, classes)
        plt.savefig('results/3_means.pdf')
        plt.close()


    lclusters.append(clusters)
    lclasses.append(classes)
    plt.plot(range(len(costs)), costs)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost History')
    plt.grid()
    plt.savefig('results/cost_%d_means.pdf' %k)
    plt.close()


plt.figure(figsize=(10,15))
for k in krange:
    plt.subplot(3,2,k, aspect='equal', adjustable='box-forced')
    plot_data(dataset, lclusters[k-1], lclasses[k-1])
plt.tight_layout()
plt.savefig('results/k_means.pdf', dpi=300)
plt.close()

np.savetxt('results/cluster_percentages.csv', percentages, fmt="%d, %.2f, %.2f, %.2f, %.2f, %.2f")
np.savetxt('results/cluster_costs.csv', val_costs, fmt="%d, %.3f")
