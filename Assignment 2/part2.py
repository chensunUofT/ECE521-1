import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import shuffle
from set_utils import make_sets, to_one_hot

# Load data
with np.load("notMNIST.npz") as data:
    images, labelso = data["images"], data["labels"]
    images = images.transpose(2,0,1)
    images = images.reshape(18720, 784)
    poissonNoise = np.random.poisson(50,784).astype(float)
    images = images.astype('float32')/255
    labels = to_one_hot(labelso)


''' PARAMETERS '''
learning_rate = 1e-2
training_epochs = 600
batch_size = 500
momentum = 1e-2
hidden_units = 1000

''' SETS '''
x_train, t_train, x_validation, t_validation, x_test, t_test = make_sets(images, labels, 15000, 1000)

#NN Model
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

#Input Layer
input_w = tf.Variable(tf.truncated_normal([784, hidden_units], stddev=0.01), name="Input_Weight")
input_b = tf.Variable(tf.truncated_normal([hidden_units], stddev=0.01), name="Input_Bias")
input_a = tf.nn.relu(tf.matmul(x, input_w) + input_b)

#Prevent overfitting with dropout
keep_prob = tf.placeholder("float")
hw_drop = tf.nn.dropout(input_a, keep_prob)

#Hidden Layer 1
hidden_w = tf.Variable(tf.truncated_normal([hidden_units, 10], stddev=0.01), name="Hidden_Weight")
hidden_b = tf.Variable(tf.truncated_normal([10], stddev=0.01), name="Hidden_Bias")
# hidden_a = tf.nn.relu(tf.matmul(hw_drop, hidden_w) + hidden_b)

# hw_drop1 = tf.nn.dropout(hidden_a1, keep_prob)
#
# #Hidden Layer 2
# hidden_w = tf.Variable(tf.truncated_normal([500, 10], stddev=0.01), name="Hidden_Weight")
# hidden_b = tf.Variable(tf.truncated_normal([10], stddev=0.01), name="Hidden_Bias")

logits = tf.add(tf.matmul(hw_drop, hidden_w), hidden_b)
hidden_a = tf.nn.softmax(logits)

#Training Specification
cost_batch = tf.nn.softmax_cross_entropy_with_logits(logits, y)
cost = tf.reduce_mean(cost_batch)

# Use momentum for training
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(cost)

# Test accuracy
correct_prediction = tf.equal(tf.argmax(hidden_a, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# Initialize variables
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

epoch = 0
costs = []
accuracies = []

# Training cycle
while epoch < training_epochs:
    epoch += 1
    total_batches = int(x_train.shape[0] / batch_size)

    # combined = zip(x_train, t_train)
    # shuffle(combined)
    # x_train[:], t_train[:] = zip(*combined)
    # Loop over all batches
    for i in range(total_batches):
        batch_xs = x_train[i*batch_size:(i+1)*batch_size]
        batch_ys = t_train[i*batch_size:(i+1)*batch_size]
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})

    train_cost = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1})
    train_accuracy = accuracy.eval({x: batch_xs, y: batch_ys, keep_prob: 1})

    validation_cost = sess.run(cost, feed_dict={x: x_validation, y: t_validation, keep_prob: 1})
    validation_accuracy = accuracy.eval({x: x_validation, y: t_validation, keep_prob: 1})

    test_cost = sess.run(cost, feed_dict={x: x_test, y: t_test, keep_prob: 1})
    test_accuracy = accuracy.eval({x: x_test, y: t_test, keep_prob: 1})

    costs.append([train_cost, validation_cost, test_cost])
    accuracies.append([train_accuracy, validation_accuracy, test_accuracy])

    if epoch % 10 == 0:
        print "Epoch: %3d" % (epoch)
        print "\tTraining Set:   Cost: %8.3f Accuracy: %d%%" %(train_cost, train_accuracy*100)
        print "\tValidation Set: Cost: %8.3f Accuracy: %d%%" %(validation_cost, validation_accuracy*100)
        print "\tTest Set:       Cost: %8.3f Accuracy: %d%%\n\n" %(test_cost, test_accuracy*100)

print "Optimization Finished!"
print "Accuracy: %5.2f%%" %(accuracy.eval({x: x_test, y: t_test, keep_prob: 1})*100)

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
plt.show()