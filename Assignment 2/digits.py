import os
import csv
import numpy as np
import tempfile
import shutil
import matplotlib.pyplot as plt
import sys
import subprocess
from neural_net import train_neural_net
from part1 import logistic_regression
from set_utils import *
from plot import plot_data
from time import time

# if not os.path.exists("models"):
#     os.makedirs("models")

if not os.path.exists("results"):
    os.makedirs("results")

with np.load("notMNIST.npz") as data:
    images, labelso = data["images"], data["labels"]
    images = images.transpose(2,0,1)
    plt.figure()
    plt.gray()
    for i in range(49):
        plt.subplot(7,7,i+1)
        plt.imshow(images[i])
        plt.axis('off')
    images = images.reshape(18720, 784)
    poissonNoise = np.random.poisson(50,784).astype(float)
    images = images.astype('float32')/255
    labels = to_one_hot(labelso)

sets = make_sets(images, labels, 15000, 1000)


print "--------------\nRUNNING PART 1\n--------------"
costs, accuracies = logistic_regression(sets)

plot_data(costs, "Cost", "part1_cost.pdf")
plot_data(accuracies, "Accuracy", "part1_accuracy.pdf")

with open("results/part1_best.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerow([costs[-1][2], accuracies[-1][2]])


print "--------------\nRUNNING PART 2\n--------------"
lr_mans = np.random.randint(1,10, size=5)
lr_logs = -np.random.randint(2,4, size=5)
lrs = (lr_mans*10.0**lr_logs).astype('float32')
mmts = lrs*5

best_cost = np.ones((1,3))*float('inf')

history = np.zeros((5, 8))

for i in range(len(lrs)):
    costs, accuracies = train_neural_net(sets, learning_rate=lrs[i], momentum=mmts[i], es=True)
    plot_data(costs, "Cost", "part2_%.4f_cost.pdf" %(lrs[i]))
    plot_data(accuracies, "Accuracy", "part2_%.4f_accuracy.pdf" %(lrs[i]))
    history[i] = np.concatenate(([lrs[i]],[mmts[i]],costs[-1],accuracies[-1]))
    if costs[-1][1] < best_cost[-1][1]:
        best_index = i
        best_cost = costs
        best_accuracy = accuracies

plot_data(best_cost, "Cost", "part2_best_cost.pdf")
plot_data(best_accuracy, "Accuracy", "part2_best_accuracy.pdf")



with open("results/part2.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(history)

with open("results/part2_best.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerow([lrs[best_index], mmts[best_index], best_cost[-1][2], best_accuracy[-1][2]])

print "--------------\nRUNNING PART 3\n--------------"
best_cost = np.ones((1,3))*float('inf')
hunits = [100,500,1000]

history = np.zeros((3, 7))

for i in range(len(hunits)):
    costs, accuracies = train_neural_net(sets, hidden_units=[hunits[i]], es=True, learning_rate=lrs[best_index])

    plot_data(costs, "Cost", "part3_%d_cost.pdf" %hunits[i])
    plot_data(accuracies, "Accuracy", "part3_%d_accuracy.pdf" %hunits[i])
    history[i] = np.concatenate(([hunits[i]],costs[-1],accuracies[-1]))
    if costs[-1][1] < best_cost[-1][1]:
        best_num = hunits[i]
        best_accuracy = accuracies
        best_cost = costs

plot_data(best_cost, "Cost", "part3_best_cost.pdf")
plot_data(best_accuracy, "Accuracy", "part3_best_accuracy.pdf")

with open("results/part3.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(history)

with open("results/part3_best.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerow([best_num, best_cost[-1][2], best_accuracy[-1][2]])



print "--------------\nRUNNING PART 4\n--------------"
costs, accuracies = train_neural_net(sets, hidden_units=[500, 500], learning_rate=lrs[best_index])

plot_data(costs, "Cost", "part4_cost.pdf")
plot_data(accuracies, "Accuracy", "part4_accuracy.pdf")

with open("results/part4_best.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerow([costs[-1][2], accuracies[-1][2]])


print "--------------\nRUNNING PART 5\n--------------"
costs, accuracies = train_neural_net(sets, hidden_units=[1000], dropout_rate=0.5, learning_rate=lrs[best_index], es=False)

plot_data(costs, "Cost", "part5_cost.pdf")
plot_data(accuracies, "Accuracy", "part5_accuracy.pdf")

with open("results/part5_best.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerow([costs[-1][2], accuracies[-1][2]])


print "--------------\nRUNNING PART 6\n--------------"
np.random.seed(int(time()))
nexp = int(raw_input("Run how many experiments? [40]") or 40)
lr_mans = np.random.randint(1,10, size=nexp/2)
lr_logs = -np.random.randint(2,5, size=nexp/2)
lrs = (lr_mans*10.0**lr_logs).astype('float32')
mmts = lrs*2
nums_layers = np.random.randint(1,4, size=nexp/2)
dropout = [1, 0.5]

results = []

for i in range(len(lrs)):
    lr = lrs[i]
    nhidden = np.random.randint(100,500, size=nums_layers[i])
    mmt = mmts[i]
    for dr in dropout:
        costs, accuracies = train_neural_net(sets, learning_rate=lr, hidden_units=nhidden, dropout_rate=dr, momentum=mmt)
        plot_data(costs, "Cost", "part6_%.4f,%s,%.2f_cost.pdf" %(lr, nhidden, dr))
        plot_data(accuracies, "Accuracy", "part6_%.4f,%s,%.2f_accuracy.pdf" %(lr, nhidden, dr))
        validation_accuracy = accuracies[-1][1]
        validation_cost = costs[-1][1]
        test_accuracy = accuracies[-1][2]
        test_cost = costs[-1][2]
        res = ["%.4f" %(lr), str(nhidden), dr, mmt, validation_cost, validation_accuracy, test_cost, test_accuracy]
        results.append(res)
        print res

results.sort(key=lambda y : y[4])
with open("results/part6.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(results)


print "----------------\nCOMPILING REPORT\n----------------"
current = os.getcwd()
temp = tempfile.mkdtemp()
shutil.copy('report.tex', temp)
shutil.move('results', temp)
os.chdir(temp)
subprocess.call(['pdflatex', 'report.tex'])
subprocess.call(['pdflatex', 'report.tex'])
shutil.copy('report.pdf', current)
shutil.rmtree(temp)
os.chdir(current)

if sys.platform == "win32":
    os.startfile('report.pdf')
else:
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, 'report.pdf'])