from load_gtsrb import load_gtsrb_images

import tensorflow as tf
import numpy as np
import cv2 as cv

import os

# configuration
logdir = os.path.dirname(os.path.realpath(__file__)) + '/log/'
datapath = os.path.dirname(os.path.realpath(__file__)) + '/GTSRB/Final_Training/Images'
chosenClasses = range(0, 43)
numClasses = len(chosenClasses)
maxSampleSize = 5000
batchSize = 512
numEpochs = 25

# reset stuff
tf.reset_default_graph()

# load data
print("Loading data...")
[images, labels, class_descs, sign_ids] = load_gtsrb_images(datapath, chosenClasses, maxSampleSize)
# change type from float to uint8 and convert to grayscale afterwards
#images = np.array([cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2GRAY) for img in images])
#images = images.reshape([len(labels), 32, 32, 1])

# split data into trainign, testing and verification
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.int32, [None])

def buildConvLayer(input, convSize, convStride, poolSize, poolStride, inDepth, outDepth):
    weights = tf.Variable(0.01*tf.random_normal([convSize,convSize,inDepth, outDepth]))
    conv = tf.nn.conv2d(input, weights, strides=convStride, padding='SAME')
    # add an bias to the
    biased = tf.nn.bias_add(conv, tf.Variable(0.01*tf.random_normal([outDepth])))
    # connect the biased convolutions to the rectified linear unit
    relu = tf.nn.relu(biased)
    # pool the relus output
    return tf.nn.max_pool(relu, ksize=[1,poolSize,poolSize,1], strides=poolStride, padding='SAME')


def buildFullyConnectedLayer(input, prevLayerSize, layerSize):
    # build mapping
    weights = tf.Variable(0.01*tf.random_normal([prevLayerSize,layerSize]))
    # MAC the input, weights and biases
    biased = tf.add(tf.matmul(input, weights), 0.01*tf.random_normal([layerSize]))
    # apply rectified linear unit
    return tf.nn.relu(biased)


# ------ build network with GPU support --------
with tf.device('/device:GPU:0'):
    convStride = [1,1,1,1]
    poolStride = [1,2,2,1]

    # add three convolutions
    first = buildConvLayer(X, 5, convStride, 2, poolStride, X.shape[3].value, 32)
    second = buildConvLayer(first, 5, convStride, 2, poolStride, 32, 64)
    lastConvLayer = buildConvLayer(second, 5, convStride, 2, poolStride, 64, 128)

    # map from convolution to neurons
    reconnection = tf.reshape(lastConvLayer, [-1,4*4*128])

    # add two fully connected layers
    fcl1 = buildFullyConnectedLayer(reconnection, 4*4*128, 2*1024)
    fcl2 = buildFullyConnectedLayer(fcl1, 2*1024, 128)

    # logits
    outweights = tf.Variable(0.01*tf.random_normal([128, numClasses]))
    out = tf.add(tf.matmul(fcl2, outweights), tf.Variable(0.01*tf.random_normal([numClasses])))

    # softmax
    pred = tf.nn.softmax(out)


# ---------- saver ------------
saver = tf.train.Saver()

# --------- optimizer ----------
crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=Y)
loss_op = tf.reduce_mean(crossentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00025)
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
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    numSamples = len(indices)
    splitPoints = [int(numSamples*7/10), int(numSamples*9/10)]
    [trainingSetIndices, validationSetIndices, testSetIndices] = np.split(indices, splitPoints)
    [trainingSetLabels, validationSetLabels, testSetLabels] = np.split(labels[indices], splitPoints)


    trainingSet = images[trainingSetIndices]
    testSet = images[testSetIndices]
    validationSet = images[validationSetIndices]

    batchesX = np.split(trainingSet, np.arange(batchSize, len(trainingSet), batchSize))
    batchesY = np.split(trainingSetLabels, np.arange(batchSize, len(trainingSetLabels), batchSize))


    # execute optimization
    print("Start training...")
    for epoch in range(0, numEpochs):

        print("Epoch: %d" % (epoch))
        for iter in range(0, len(batchesX)):
            sess.run(train_op, feed_dict={X: batchesX[iter], Y: batchesY[iter]})
            print("Batch %d/%d" % (iter, len(batchesX)-1))

        saver.save(sess, logdir, global_step=epoch)

        loss = sess.run(loss_op, feed_dict={X: trainingSet[:1000], Y: trainingSetLabels[:1000]})
        print("Test Loss: %f" % (loss))

        loss = sess.run(loss_op, feed_dict={X: validationSet[:1000], Y: validationSetLabels[:1000]})
        print("Validation Loss: %f" % (loss))

    print("Optimization done!")

    # check performance
    result = pred.eval( {X: testSet[:1000]} )
    predictions = tf.equal(tf.argmax(result, 1), tf.cast(testSetLabels[:1000], tf.int64)).eval()
    print("Test accuracy: %s" % (np.sum(predictions)/len(predictions)))
