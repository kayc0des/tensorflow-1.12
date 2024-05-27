#!/usr/bin/env python3
''' low level tensorflow '''


import tensorflow as tf


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


# debug
x, y = create_placeholders(784, 10)
print(x)
print(y)
l = create_layer(x, 256, tf.nn.tanh)
print(l)
