from k_means import k_means, plot_data, get_cost
from mpl_toolkits.mplot3d import Axes3D
from mixture_of_gaussians import train_mog_model, plot_mog
import numpy as np
import os
import matplotlib.pyplot as plt

def normalize(data):
    data -= np.mean(data)
    return data

# Create Results Dir
if not os.path.exists("results"):
    os.makedirs("results")

# Load datasets
data = np.load('data2D.npy').astype('float32')
data2 = np.load('data100D.npy').astype('float32')
points, points2 = len(data), len(data2)
data = normalize(data)
data2 = normalize(data2)
validation, validation2 = data[:points/3], data2[:points2/3]
dataset, dataset2 = data[points/3:], data2[points2/3:]
#
# '''
# PART 1.2.1
# '''
# #vals2 = evaluate_cost_function()
# costs = np.load('costs.npy')
# fig = plt.figure()
# ax = fig.add_axes([.05, .05, .9, .9], projection='3d')
# ax.plot_trisurf(costs[::5,0], costs[::5,1], costs[::5,2], cmap=plt.cm.Spectral, shade=False, linewidth=0.1)
# ax.set_xlabel('Cluster 1')
# ax.set_ylabel('Cluster 2')
# ax.set_zlabel('Cost')
# plt.grid()
# plt.savefig('results/cost_function.pdf', dpi=50)
# plt.close()
#
#
# '''
# PARTS 1.2.2, 1.2.3, 1.2.4
# '''
# krange = range(1,6)
# percentages = np.zeros((len(krange), max(krange)+1))
# val_costs = np.zeros((len(krange), 2))
# lclusters, lclasses = [], []
# for k in krange:
#     clusters, classes, costs = k_means(dataset, k)
#
#     percentages[k-1][0] = k
#     for kclass in range(k):
#         percentages[k-1][kclass+1] = np.sum(np.equal(classes, kclass)) * 100.0/len(classes)
#
#     val_costs[k-1] = np.array([k, get_cost(validation, clusters)])
#
#     # Make separate plot of clusters when K = 3
#     if k == 3:
#         plot_data(dataset, clusters, classes)
#         plt.savefig('results/3_means.pdf')
#         plt.close()
#         plt.plot(range(len(costs)), costs)
#         plt.xlabel('Iteration')
#         plt.ylabel('Cost')
#         plt.title('K-Means Training Curve')
#         plt.grid()
#         plt.savefig('results/cost_3_means.pdf')
#
#
#     lclusters.append(clusters)
#     lclasses.append(classes)
#
# plt.plot(val_costs[:,0], val_costs[:,1])
# plt.xlabel('Number of Clusters')
# plt.ylabel('Cost')
# plt.title('Cost History')
# plt.grid()
# plt.savefig('results/kmeans_2d_ksweep.pdf')
# plt.close()
#
#
# plt.figure(figsize=(10,15))
# for k in krange:
#     plt.subplot(3,2,k, aspect='equal', adjustable='box-forced')
#     plot_data(dataset, lclusters[k-1], lclasses[k-1])
# plt.tight_layout()
# plt.savefig('results/k_means.pdf', dpi=300)
# plt.close()
#
# np.savetxt('results/cluster_percentages.csv', percentages, fmt="%d, %.2f, %.2f, %.2f, %.2f, %.2f")
# np.savetxt('results/cluster_costs.csv', val_costs, fmt="%d, %.3f")
#
#
# '''
# PART 2.2.2
# '''
#
# (pz, mu, sigma), assignments, fcost,_ = train_mog_model(dataset, 3, validation, 3)
#
# with open('results/mog_2d_parameters.csv', 'w+') as f:
#     for i in range(3):
#         f.write('%.3f; [%.3f; %.3f]; %.3f\n' %(pz[0,i], mu[i,0],mu[i,1], sigma[i,0]))
#
# plot_mog(validation, mu, np.sqrt(sigma), assignments)
# plt.savefig('results/mog_2d.pdf')
# plt.close()
#
# plt.plot(range(len(fcost)), fcost)
# plt.xlabel("Iteration")
# plt.ylabel("Cost")
# plt.title("MoG Training Curve")
# plt.grid()
# plt.savefig('results/mog_2d_train.pdf')
# plt.close()


#
# '''
# PART 2.2.3
# '''
# costs = []
# plt.figure(figsize=(10,15))
# for k in range(1,6):
#     (pz, mu, sigma), assignments, _, vcost = train_mog_model(dataset, k, validation, 3)
#     costs.append([k,vcost])
#     ax = plt.subplot(3,2,k)
#     plot_mog(validation, mu, np.sqrt(sigma), assignments, ax)
# plt.tight_layout()
# plt.savefig('results/mogs.pdf', dpi=300)
# plt.close()
# costs = np.array(costs)
# np.savetxt('results/mog_costs.csv', costs, fmt="%d, %.3f")
#
# plt.plot(costs[:,0],costs[:,1])
# plt.xlabel("Number of Clusters")
# plt.ylabel("Log Likelihood")
# plt.title("MoG Clusters")
# plt.grid()
# plt.savefig('results/mog_2d_ksweep.pdf')
# plt.close()


'''
PART 2.2.4
'''
costs = []
for k in range(1,10):
    # vcost = train_mog_model(dataset2, k, validation2, 3)[-1]
    clusters = k_means(dataset2, k)[0]
    kvcost = get_cost(validation2, clusters)
    costs.append([k,0,kvcost])
    # print "%d - %.3f - %.3f" %(k, vcost, kvcost)

# costs = np.array(costs)
# np.savetxt('results/costs100d.csv', costs, fmt="%d, %.3f, %.3f")
#
# plt.plot(costs[:,0],costs[:,1])
# plt.xlabel("Number of Clusters")
# plt.ylabel("Log Likelihood")
# plt.title("MoG Clusters")
# plt.grid()
# plt.savefig('results/mog_100d.pdf')
# plt.close()

costs = np.array(costs)
print costs[:,2]
plt.plot(costs[:,0],costs[:,2])
plt.xlabel("Number of Clusters")
plt.ylabel("Squared Error")
plt.title("K-Means Clusters")
plt.grid()
plt.savefig('results/k_means_100d.pdf')
plt.close()