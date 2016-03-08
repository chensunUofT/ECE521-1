import numpy as np
import tensorflow as tf
import time, datetime
import os
import re

def train_neural_net(sets, hidden_units=[1000], dropout_rate=1, learning_rate=1e-2, momentum=None, batch_size=100, training_epochs=500, es=False):
    if momentum is None:
        momentum = float(learning_rate*5)

    if not os.path.exists('models/%s,%s' %(hidden_units, dropout_rate)):
        os.makedirs('models/%s,%s' %(hidden_units, dropout_rate))

    train_config = {'learning rate':learning_rate, 'momentum': momentum, 'batch size':batch_size,\
                    'epochs':training_epochs, 'dropout': dropout_rate}
    topology = {'units per layer':hidden_units}

    ''' SETS '''
    x_train, t_train, x_validation, t_validation, x_test, t_test = sets

    #NN Model
    x = tf.placeholder(tf.float32, [None, 784], name="Inputs")
    y = tf.placeholder(tf.float32, [None, 10], name="Expected_Ouputs")
    keep_prob = tf.placeholder("float", name="Dropout_Rate")

    layer_w, layer_b, layer_a, layer_drop = [], [], [], []
    hist_w, hist_b = [], []

    #Input Layer
    with tf.name_scope("Input_Layer") as scope:
        layer_w.append(tf.Variable(tf.truncated_normal([784, hidden_units[0]], stddev=0.01), name="Input_Weight"))
        tf.histogram_summary("Input_Weight", layer_w[0])
        layer_b.append(tf.Variable(tf.truncated_normal([hidden_units[0]], stddev=0.01), name="Input_Bias"))
        tf.histogram_summary("Input_Bias", layer_b[0])
        layer_a.append(tf.nn.relu(tf.matmul(x, layer_w[0]) + layer_b[0]))
        #Prevent overfitting with dropout
        layer_drop.append(tf.nn.dropout(layer_a[0], keep_prob))


    for i in range(1,len(hidden_units)):
        with tf.name_scope("Hidden_Layer_%d" %(i)) as scope:
            layer_w.append(tf.Variable(tf.truncated_normal([hidden_units[i-1], hidden_units[i]], stddev=0.01), name="Hidden_Weight_%d" %(i)))
            tf.histogram_summary("Hidden_Weight_%d" %(i), layer_w[i])
            layer_b.append(tf.Variable(tf.truncated_normal([hidden_units[i]], stddev=0.01), name="Hidden_Bias_%d" %(i)))
            tf.histogram_summary("Hidden_Bias_%d" %(i), layer_b[i])
            layer_a.append(tf.nn.relu(tf.matmul(layer_drop[i-1], layer_w[i]) + layer_b[i]))
            layer_drop.append(tf.nn.dropout(layer_a[i], keep_prob))


    with tf.name_scope("Output_Layer") as scope:
        layer_w.append(tf.Variable(tf.truncated_normal([hidden_units[-1], 10], stddev=0.01), name="Output_Weight"))
        tf.histogram_summary("Output_Weight", layer_w[-1])
        layer_b.append(tf.Variable(tf.truncated_normal([10], stddev=0.01), name="Output_Bias"))
        tf.histogram_summary("Output_Bias", layer_b[-1])
        logits = tf.add(tf.matmul(layer_drop[-1], layer_w[-1]), layer_b[-1])
        layer_a.append(tf.nn.softmax(logits))
    output_a = layer_a[-1]


    #Training Specification
    with tf.name_scope("Train") as scope:
        cost_batch = tf.nn.softmax_cross_entropy_with_logits(logits, y)
        cost = tf.reduce_mean(cost_batch)
        train_cost_summary = tf.scalar_summary("Training_Cost", cost)
        validation_cost_summary = tf.scalar_summary("Validation_Cost", cost)
        test_cost_summary = tf.scalar_summary("Test_Cost", cost)
        # Use momentum for training
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(cost)

    # Test accuracy
    with tf.name_scope("Output_Accuracy") as scope:
        correct_prediction = tf.equal(tf.argmax(output_a, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100
        train_accuracy_summary = tf.scalar_summary("Training_Accuracy", accuracy)
        validation_accuracy_summary = tf.scalar_summary("Validation_Accuracy", accuracy)
        test_accuracy_summary = tf.scalar_summary("Test_Accuracy", accuracy)

    sess = tf.Session()
    epoch = 0
    costs = []
    accuracies = []

    init = tf.initialize_all_variables()
    sess.run(init)

    #Check if there is a model saved, if there is, load it.
    # saver = tf.train.Saver()
    # ckpt = tf.train.get_checkpoint_state('models/%s,%s/' %(hidden_units, dropout_rate))
    # if ckpt and raw_input('Found model saved on disk, load? [Y/n]').lower() != 'n':
    #     ckpt_file = ckpt.model_checkpoint_path
    #     saver.restore(sess, ckpt_file)
    #     epoch = int(re.findall(r'\d+$', ckpt_file)[-1])

        # Initialize variables
    # merged = tf.merge_all_summaries()
    # writer = tf.train.SummaryWriter('tensorboard/%s,%s/' %(hidden_units, dropout_rate), sess.graph_def)


    print "------------------------------------"
    print train_config
    print topology
    print "------------------------------------"

    with sess.as_default():
        # Training cycle
        while epoch < training_epochs:
            epoch += 1
            total_batches = int(x_train.shape[0] / batch_size)

            # combined = zip(x_train, t_train)
            # np.random.shuffle(combined)
            # x_train[:], t_train[:] = zip(*combined)
            # Loop over all batches
            for i in range(total_batches):
                batch_xs = x_train[i*batch_size:(i+1)*batch_size]
                batch_ys = t_train[i*batch_size:(i+1)*batch_size]
                # Fit training using batch data
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_rate})

            # t_acc_sum, t_cost_sum, train_accuracy, train_cost = \
            #     sess.run([train_accuracy_summary, train_cost_summary, accuracy, cost], \
            #              feed_dict={x: x_train, y: t_train, keep_prob: 1})
            # writer.add_summary(t_acc_sum, epoch)
            # writer.add_summary(t_cost_sum, epoch)
            #
            # v_acc_sum, v_cost_sum, validation_accuracy, validation_cost = \
            #     sess.run([validation_accuracy_summary, validation_cost_summary, accuracy, cost], \
            #              feed_dict={x: x_validation, y: t_validation, keep_prob: 1})
            # writer.add_summary(v_acc_sum, epoch)
            # writer.add_summary(v_cost_sum, epoch)
            #
            # te_acc_sum, te_cost_sum, test_accuracy, test_cost = \
            #     sess.run([test_accuracy_summary, test_cost_summary, accuracy, cost], \
            #              feed_dict={x: x_test, y: t_test, keep_prob: 1})
            # writer.add_summary(te_acc_sum, epoch)
            # writer.add_summary(te_cost_sum, epoch)

            train_cost = sess.run(cost, feed_dict={x: x_train, y: t_train, keep_prob: 1})
            train_accuracy = accuracy.eval({x: x_train, y: t_train, keep_prob: 1})

            validation_cost = sess.run(cost, feed_dict={x: x_validation, y: t_validation, keep_prob: 1})
            validation_accuracy = accuracy.eval({x: x_validation, y: t_validation, keep_prob: 1})

            test_cost = sess.run(cost, feed_dict={x: x_test, y: t_test, keep_prob: 1})
            test_accuracy = accuracy.eval({x: x_test, y: t_test, keep_prob: 1})

            costs.append([train_cost, validation_cost, test_cost])
            accuracies.append([train_accuracy, validation_accuracy, test_accuracy])

            if epoch % 10 == 0:
                st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
                print "[%s] Epoch: %3d" % (st, epoch)
                print "\tTraining Set:   Cost: %8.3f Accuracy: %d%%" %(train_cost, train_accuracy)
                print "\tValidation Set: Cost: %8.3f Accuracy: %d%%" %(validation_cost, validation_accuracy)
                print "\tTest Set:       Cost: %8.3f Accuracy: %d%%\n\n" %(test_cost, test_accuracy)
                # if epoch % 50 == 0:
                #     saver.save(sess, 'models/%s,%s/nn.ckpt' %(hidden_units, dropout_rate), global_step=epoch)
                if es and validation_cost > 1.01*costs[-10][1]:
                    print "Early stopping!"
                    break
                if abs(train_cost - costs[-10][0]) < 1e-5:
                    print "Converged!"
                    break

        print "Optimization Finished!"

        train_cost = sess.run(cost, feed_dict={x: x_train, y: t_train, keep_prob: 1})
        train_accuracy = accuracy.eval({x: x_train, y: t_train, keep_prob: 1})

        validation_cost = sess.run(cost, feed_dict={x: x_validation, y: t_validation, keep_prob: 1})
        validation_accuracy = accuracy.eval({x: x_validation, y: t_validation, keep_prob: 1})

        test_cost = sess.run(cost, feed_dict={x: x_test, y: t_test, keep_prob: 1})
        test_accuracy = accuracy.eval({x: x_test, y: t_test, keep_prob: 1})

        costs.append([train_cost, validation_cost, test_cost])
        accuracies.append([train_accuracy, validation_accuracy, test_accuracy])

    costs = np.array(costs)

    accuracies = np.array(accuracies)

    return costs, accuracies