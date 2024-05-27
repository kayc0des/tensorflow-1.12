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
    
    # initialize dictionarie's to hold values of z and a
    output_z = {}
    ouput_a = {}
    layers = {}

    with tf.Session as sess:
        for i in range(len(layer_sizes):
                if i == 0:
                    layer[i + 1] = create_layer(x, layer_sizes[i], activations[i])     
                    output_z['z' + str(i + 1)] = tf.add(tf.matmul(x, layer[i + 1].get_variable('dense/kernel')))
                    output_a['a' + str(i + 1)] = activations[i](output['z' + str(i + 1)])
                else:
                    # to do

    


# debug
x, y = create_placeholders(784, 10)
print(x)
print(y)
l = create_layer(x, 256, tf.nn.tanh)
print(l)































