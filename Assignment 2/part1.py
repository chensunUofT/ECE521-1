import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import string
from set_utils import make_sets, to_one_hot

# Load data
with np.load("notMNIST.npz") as data:
    images, labels = data["images"], data["labels"]
    images = images.transpose(2,0,1)
    images = images.reshape(18720, 784)
    images /= 255
    labels = to_one_hot(labels)

''' PARAMETERS '''
learning_rate = 1e-2
training_epochs = 400
batch_size = 1000
momentum = 1e-4

''' SETS '''
x_train, t_train, x_validation, t_validation, x_test, t_test = make_sets(images, labels, 15000, 1000)

''' MODEL '''
# Tensorflow Placeholders
x = tf.placeholder(tf.float32, [None, 784], name="Input") # Images have shape 28x28 = 784
y = tf.placeholder(tf.float32, [None, 10], name="Class") # 10 classes

# TensorFlow Variables
W = tf.Variable(tf.random_normal([784, 10], stddev=0.001), name="Weight")
b = tf.Variable(tf.zeros([10]), name="Bias")

# Output layer is softmax
logits = tf.add(tf.matmul(x, W), b)
activation = tf.nn.softmax(logits)

# Log Likelihood
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))

# Use momentum for training
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(cost)

correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# Initialize variables
sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)

with sess.as_default():
    curr_acc = accuracy.eval({x: x_validation, y: t_validation})
    epoch = 0
    costs = []
    accuracies = []

    # Training cycle
    while epoch < training_epochs:
        epoch += 1
        total_batches = int(x_train.shape[0] / batch_size)
        # Loop over all batches
        for i in range(total_batches):
            batch_xs = x_train[i*batch_size:(i+1)*batch_size]
            batch_ys = t_train[i*batch_size:(i+1)*batch_size]
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

        train_cost = sess.run(cost, feed_dict={x: x_train, y: t_train})
        train_accuracy = accuracy.eval({x: x_train, y: t_train})

        validation_cost = sess.run(cost, feed_dict={x: x_validation, y: t_validation})
        validation_accuracy = accuracy.eval({x: x_validation, y: t_validation})

        test_cost = sess.run(cost, feed_dict={x: x_test, y: t_test})
        test_accuracy = accuracy.eval({x: x_test, y: t_test})

        costs.append([train_cost, validation_cost, test_cost])
        accuracies.append([train_accuracy, validation_accuracy, test_accuracy])

        if epoch % 10 == 0:
            print "Epoch: %3d" % (epoch)
            print "\tTraining Set:   Cost: %8.3f Accuracy: %d%%" %(train_cost, train_accuracy*100)
            print "\tValidation Set: Cost: %8.3f Accuracy: %d%%" %(validation_cost, validation_accuracy*100)
            print "\tTest Set:       Cost: %8.3f Accuracy: %d%%\n\n" %(test_cost, test_accuracy*100)

    print "Optimization Finished!"
    print "Accuracy: %5.2f%%" %(accuracy.eval({x: x_test, y: t_test})*100)

costs = np.array(costs)
costs /= [len(x_train), len(x_validation), len(x_test)]

accuracies = np.array(accuracies)
accuracies *= 100

plt.figure()
plt.plot(range(epoch), costs[:,0], label="Training Set")
plt.plot(range(epoch), costs[:,1], label="Validation Set")
plt.plot(range(epoch), costs[:,2], label="Test Set")
plt.legend(loc='best')
plt.title("Cost History")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.grid()

plt.figure()
plt.plot(range(epoch), accuracies[:,0], label="Training Set")
plt.plot(range(epoch), accuracies[:,1], label="Validation Set")
plt.plot(range(epoch), accuracies[:,2], label="Test Set")
plt.legend(loc='best')
plt.title("Accuracy History")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()

plt.figure()
ws = W.eval()
plt.suptitle("Weight Visualization")
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.imshow(ws[:,i].reshape(28,28))
    plt.title(string.ascii_uppercase[i])
plt.show()
