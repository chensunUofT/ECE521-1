import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from plot import visualize_weights

def logistic_regression(sets, learning_rate=1e-2, training_epochs=500, batch_size=100, momentum=5e-2, es=True):

    ''' SETS '''
    x_train, t_train, x_validation, t_validation, x_test, t_test = sets

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
                if es and validation_cost > 1.01*costs[-10][1]:
                    print "Early stopping!"
                    break

        print "Optimization Finished!"
        print "Accuracy: %5.2f%%" %(accuracy.eval({x: x_test, y: t_test})*100)

    costs = np.array(costs)

    accuracies = np.array(accuracies)
    accuracies *= 100

    plt.figure()
    ws = W.eval()
    visualize_weights(ws, "part1_weights.pdf")

    return costs, accuracies
