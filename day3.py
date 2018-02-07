from load_gtsrb import load_gtsrb_images

import tensorflow as tf
import numpy as np
import cv2 as cv

import os

# configuration
logdir = os.path.dirname(os.path.realpath(__file__)) + '/log/'
datapath = os.path.dirname(os.path.realpath(__file__)) + '/GTSRB/Final_Training/Images'
chosenClasses = range(0, 10)
#chosenClasses = range(0,43)
numClasses = len(chosenClasses)
maxSampleSize = 500
batchSize = 15

# reset stuff
tf.reset_default_graph()

# load data
print("Loading data...")
[images, labels, class_descs, sign_ids] = load_gtsrb_images(datapath, chosenClasses, maxSampleSize)
# change type from float to uint8 and convert to grayscale afterwards
greyscaleImages = np.array([cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2GRAY) for img in images])

# split data into trainign, testing and verification
X = tf.placeholder(tf.float32, [None, 32, 32])
Y = tf.placeholder(tf.int32, [None])

# ------ build network with GPU support --------
with tf.device('/device:GPU:0'):
    xInput = tf.reshape(X, shape=[-1,32,32,1])

    # first convolution
    weights = tf.Variable(tf.random_normal([5,5,1,32]))
    conv = tf.nn.conv2d(xInput, weights, strides=[1,1,1,1], padding='SAME')
    relu = tf.nn.relu(conv)
    mpool = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # from convolution to neurons
    reconnection = tf.reshape(mpool, [-1,16*16*32])
    wd1 = tf.Variable(tf.random_normal([16*16*32,64]))
    fcl = tf.matmul(reconnection, wd1)

    # fully connected layer
    outweights = tf.Variable(tf.random_normal([64, numClasses]))
    fcl_relu = tf.nn.relu(fcl)
    out = tf.matmul(fcl_relu, outweights)

    # softmax
    pred = tf.nn.softmax(out)


# ---------- saver ------------
saver = tf.train.Saver()

# --------- optimizer ----------
crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=Y)
loss_op = tf.reduce_mean(crossentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

# ------- launch session -------
with tf.Session() as sess:
    run_metadata = tf.RunMetadata()

    # init session
    sess.run(tf.global_variables_initializer())

    # restore last session
    #saver.restore(sess, tf.train.latest_checkpoint(logdir))

    # save graph
    train_writer = tf.summary.FileWriter(logdir+'/train', sess.graph)
    train_writer.close()

    #build batches
    print("Generating test and training sets...")
    indices = np.arange(len(greyscaleImages))
    np.random.shuffle(indices)

    [trainingSetIndices, testSetIndices, validationSetIndices] = np.array_split(indices, 3)
    [trainingSetLabels, testSetLabels, validationSetLabels] = np.array_split(labels[indices], 3)

    trainingSet = greyscaleImages[trainingSetIndices]
    testSet = greyscaleImages[testSetIndices]
    validationSet = greyscaleImages[validationSetIndices]

    print("Start training...")
    batchesX = np.split(trainingSet, np.arange(batchSize, len(trainingSet), batchSize))
    batchesY = np.split(trainingSetLabels, np.arange(batchSize, len(trainingSetLabels), batchSize))
    # execute optimization
    for iter in range(0, len(batchesX)):
        sess.run(train_op, feed_dict={X: batchesX[iter], Y: batchesY[iter]})
        saver.save(sess, logdir, global_step=iter)
        loss = sess.run(loss_op, feed_dict={X: batchesX[iter], Y: batchesY[iter]})
        print("Loss: %f" % loss)

    print("Optimization done!")

    # check performance
    result = pred.eval( {X: testSet[:256]} )
    predictions = tf.equal(tf.argmax(result, 1), tf.cast(testSetLabels[:256], tf.int64)).eval()
    print("Test accuracy %s:" % (np.sum(predictions)/len(predictions)))
