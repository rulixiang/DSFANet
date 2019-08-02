# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import logging

logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

def dsfa(xtrain, ytrain, xtest, ytest, net_shape=None, args=None):

    train_num = np.shape(xtrain)[0]
    bands = np.shape(xtrain)[-1]

    tf.reset_default_graph()

    activation = tf.nn.softsign

    xd = tf.placeholder(dtype=tf.float32, shape=[None, bands])
    yd = tf.placeholder(dtype=tf.float32, shape=[None, bands])

    # fc1
    fc1w1 = tf.Variable(tf.truncated_normal(shape=[bands, net_shape[0]], dtype=tf.float32, stddev=1e-1))
    fc1w2 = tf.Variable(tf.truncated_normal(shape=[bands, net_shape[0]], dtype=tf.float32, stddev=1e-1))
    fc1b1 = tf.Variable(tf.constant(1e-1, shape=[net_shape[0]], dtype=tf.float32))
    fc1b2 = tf.Variable(tf.constant(1e-1, shape=[net_shape[0]], dtype=tf.float32))

    fc1x = tf.nn.bias_add(tf.matmul(xd, fc1w1), fc1b1)
    fc1y = tf.nn.bias_add(tf.matmul(yd, fc1w2), fc1b2)

    fc11 = activation(fc1x)
    fc12 = activation(fc1y)

    # fc2
    fc2w1 = tf.Variable(tf.truncated_normal(shape=[net_shape[0], net_shape[1]], dtype=tf.float32, stddev=1e-1))
    fc2w2 = tf.Variable(tf.truncated_normal(shape=[net_shape[0], net_shape[1]], dtype=tf.float32, stddev=1e-1))
    fc2b1 = tf.Variable(tf.constant(1e-1, shape=[net_shape[1]], dtype=tf.float32))
    fc2b2 = tf.Variable(tf.constant(1e-1, shape=[net_shape[1]], dtype=tf.float32))

    fc2x = tf.nn.bias_add(tf.matmul(fc11, fc2w1), fc2b1)
    fc2y = tf.nn.bias_add(tf.matmul(fc12, fc2w2), fc2b2)

    fc21 = activation(fc2x)
    fc22 = activation(fc2y)

    # fc3
    fc3w1 = tf.Variable(tf.truncated_normal(shape=[net_shape[1], net_shape[2]], dtype=tf.float32, stddev=1e-1))
    fc3w2 = tf.Variable(tf.truncated_normal(shape=[net_shape[1], net_shape[2]], dtype=tf.float32, stddev=1e-1))
    fc3b1 = tf.Variable(tf.constant(1e-1, shape=[net_shape[2]], dtype=tf.float32))
    fc3b2 = tf.Variable(tf.constant(1e-1, shape=[net_shape[2]], dtype=tf.float32))

    fc3x = tf.nn.bias_add(tf.matmul(fc21, fc3w1), fc3b1)
    fc3y = tf.nn.bias_add(tf.matmul(fc22, fc3w2), fc3b2)

    #fc3x = activation(fc3x)
    #fc3y = activation(fc3y)

    fc_x = fc3x
    fc_y = fc3y

    Differ = fc_x - fc_y

    A = tf.matmul(Differ, Differ, transpose_a=True)
    A = A / train_num

    sigmaX = tf.matmul(fc_x, fc_x, transpose_a=True) #+ lamb * tf.eye(net_shape[-1])
    sigmaY = tf.matmul(fc_y, fc_y, transpose_a=True) #+ lamb * tf.eye(net_shape[-1])
    sigmaX = sigmaX / train_num
    sigmaY = sigmaY / train_num

    B = (sigmaX + sigmaY) / 2 + args.reg * tf.eye(net_shape[-1])
    sigma = tf.matmul(tf.linalg.inv(B), A)+ args.reg * tf.eye(net_shape[-1])
    D, V = tf.linalg.eigh(sigma)
    
    loss = tf.sqrt(tf.reduce_sum(tf.trace(tf.matmul(sigma,sigma))))

    optimizer = tf.train.GradientDescentOptimizer(args.lr).minimize(loss)

    init = tf.global_variables_initializer()

    loss_log = []

    gpu_options = tf.GPUOptions(allow_growth = True)
    conf        = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=conf)

    sess.run(init)
    #writer = tf.summary.FileWriter('graph')
    #writer.add_graph(sess.graph)

    for k in range(args.epoch):
        sess.run(optimizer, feed_dict={xd: xtrain, yd: ytrain})

        if k % 100 == 0:
            ll = sess.run(loss, feed_dict={xd: xtrain, yd: ytrain})
            ll = ll / net_shape[-1]
            logging.info('The %4d-th epochs, loss is %4.4f ' % (k, ll))
            loss_log.append(ll)

    matV = sess.run(V, feed_dict={xd: xtest, yd: ytest})
    bVal = sess.run(B, feed_dict={xd: xtest, yd: ytest})

    fcx = sess.run(fc_x, feed_dict={xd: xtest, yd: ytest})
    fcy = sess.run(fc_y, feed_dict={xd: xtest, yd: ytest})

    sess.close()
    print('')

    return loss_log, matV, fcx, fcy, bVal
