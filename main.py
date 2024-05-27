#!/usr/bin/env python3
''' low level tensorflow '''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_placeholders(nx, classes):
    '''
    Create placeholders

    Param:
    nx -> number of feature columns in our data = 784
    classes -> number of classes in our classifier = 10

    retuns:
    x and y = tf.tensor objects
    '''

    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')

    return x, y


def create_layer(prev, n, activation):
    '''
    Create layer

    Param:
    prev -> tensor output of the prev layer
    n -> number of nodes in the layer to create
    activation -> activation funvtion to be used by the layer

    Returns:
    The tensor output of the layer
    '''
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation, kernel_initializer=weights ,name='layer')
    
    output = layer(prev)

    return output


def forward_prop(x, layer_sizes=[], activations=[]):
    '''
    Creates the forward propagation graph

    Param:
    x -> placeholder fot the input data
    layer_sizes -> list containing the number of nodes of the network
    activations -> list containing activation functions for each layer of the network

    Returns:
    Prediction for the network in tensor form
    '''

    output = x

    for size, activation in zip(layer_sizes, activations):
        output = create_layer(output, size, activation)
    
    return output


def calculate_accuracy(y, y_pred):
    '''
    Calculate accuracy
    '''

    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1)), tf.float32))


def calculate_loss(y, y_pred):
    '''
        that calculates the loss of a prediction:
    '''
    return tf.losses.softmax_cross_entropy(y, y_pred)


def create_train_op(loss, alpha):
    '''
        that creates the training
        operation for the network:
    '''
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier

    returns:
        path to where model was saved
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations):
            loss_train = sess.run(loss,
                                  feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            loss_valid = sess.run(loss,
                                  feed_dict={x: X_valid, y: Y_valid})
            accuracy_valid = sess.run(accuracy,
                                      feed_dict={x: X_valid, y: Y_valid})
            if (i % 100) is 0:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(loss_train))
                print("\tTraining Accuracy: {}".format(accuracy_train))
                print("\tValidation Cost: {}".format(loss_valid))
                print("\tValidation Accuracy: {}".format(accuracy_valid))
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        i += 1
        loss_train = sess.run(loss,
                              feed_dict={x: X_train, y: Y_train})
        accuracy_train = sess.run(accuracy,
                                  feed_dict={x: X_train, y: Y_train})
        loss_valid = sess.run(loss,
                              feed_dict={x: X_valid, y: Y_valid})
        accuracy_valid = sess.run(accuracy,
                                  feed_dict={x: X_valid, y: Y_valid})
        print("After {} iterations:".format(i))
        print("\tTraining Cost: {}".format(loss_train))
        print("\tTraining Accuracy: {}".format(accuracy_train))
        print("\tValidation Cost: {}".format(loss_valid))
        print("\tValidation Accuracy: {}".format(accuracy_valid))
        return saver.save(sess, save_path)

def evaluate(X, Y, save_path):
    """
    Evaluates output of neural network

    parameters:
        X [numpy.ndarray]: contains the input data to evaluate
        Y [numpy.ndarray]: contains the one-hot labels for X
        save_path [string]: location to load the model from

    returns:
        the network's prediction, accuracy, and loss, respectively
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss = sess.run(loss, feed_dict={x: X, y: Y})
    return (prediction, accuracy, loss)


def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('data/MNIST.npz')
    X_test_3D = lib['X_test']
    Y_test = lib['Y_test']
    X_test = X_test_3D.reshape((X_test_3D.shape[0], -1))
    Y_test_oh = one_hot(Y_test, 10)

    Y_pred_oh, accuracy, cost = evaluate(X_test, Y_test_oh, './model.ckpt')
    print("Test Accuracy:", accuracy)
    print("Test Cost:", cost)

    Y_pred = np.argmax(Y_pred_oh, axis=1)

    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_test_3D[i])
        plt.title(str(Y_test[i]) + ' : ' + str(Y_pred[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

