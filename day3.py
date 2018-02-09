from load_gtsrb import load_gtsrb_images
from util import pretty_time_delta, batchify

import tensorflow as tf
import numpy as np

import os
import time
import datetime
import argparse

# define some command line args to simply change runtime parameters
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to run.')
parser.add_argument('--learning_rate', type=float, default=0.00025, help='Initial learning rate')
parser.add_argument('--dropout', type=float, default=0.85, help='Keep probability for training dropout.')
parser.add_argument('--introspect', action='store_true', help='Generate additional data to introspect the network.')
parser.add_argument('--validation_freqency', type=int, default=10, help='Number of steps between two validations when learning.')
parser.add_argument('--max_samples_per_class', type=int, default=100, help='Maximal number of samples per class.')
parser.add_argument('--batch_size', type=int, default=100, help='Number of samples per batch for training.')
parser.add_argument('--logdir', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log/'), help='Directory for logging tensorflow data.')
parser.add_argument('--datadir', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'GTSRB/Final_Training/Images/'), help='Directory with the actual image files.')
args = parser.parse_args()

# configuration
logdir = args.logdir
datadir = args.datadir
maxSampleSize = args.max_samples_per_class
batchSize = args.batch_size
numEpochs = args.epochs
pdropout = args.dropout
learning_rate = args.learning_rate
validationFreqency = args.validation_freqency
introspect = args.introspect

convStride = [1,1,1,1]
poolStride = [1,2,2,1]
chosenClasses = range(43)
numClasses = len(chosenClasses)

# reset stuff
tf.reset_default_graph()

# load data
print("Loading data...")
np.random.seed(seed=1337)
[images, labels, class_descs, sign_ids] = load_gtsrb_images(datadir, chosenClasses, maxSampleSize)
# generate indices here, so they directly depend on the random seed an are therefore reproducible trough multiple runs, such that the sets are deterministically selected
indices = np.arange(len(images))
np.random.shuffle(indices)

global_step = tf.Variable(0, name='global_step', trainable=False)

# graph input
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    Y = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(tf.float32)


def buildConvLayer(input, convSize, convStride, poolSize, poolStride, inDepth, outDepth):
    # generate variables
    weights = tf.Variable(0.01*tf.random_normal([convSize,convSize,inDepth, outDepth]))
    biases = tf.Variable(0.01*tf.random_normal([outDepth]))

    # add actual convolution
    conv = tf.nn.conv2d(input, weights, strides=convStride, padding='SAME')
    # add an bias to the
    biased = tf.nn.bias_add(conv, biases)
    # connect the biased convolutions to the rectified linear unit
    relu = tf.nn.relu(biased)
    # pool the relus output
    pooling = tf.nn.max_pool(relu, ksize=[1,poolSize,poolSize,1], strides=poolStride, padding='SAME')

    tf.summary.histogram('weight_distribution', weights, collections=['per_epoch'])
    tf.summary.histogram('bias_distribution', biases, collections=['per_epoch'])
    tf.summary.histogram('relu_activation', relu, collections=['per_epoch'])

    return [pooling, weights, biases]


def buildFullyConnectedLayer(input, prevLayerSize, layerSize, dropout):
    # build mapping
    weights = tf.Variable(0.01*tf.random_normal([prevLayerSize,layerSize]))
    biases = 0.01*tf.random_normal([layerSize])

    # MAC the input, weights and biases
    biased = tf.add(tf.matmul(input, weights), biases)
    # apply rectified linear unit
    relu = tf.nn.relu(biased)

    tf.summary.histogram("weight_distribution", weights, collections=['per_epoch'])
    tf.summary.histogram("bias_distribution", biases, collections=['per_epoch'])
    tf.summary.histogram('relu_activation', relu, collections=['per_epoch'])

    return tf.nn.dropout(relu, dropout)


# build network with GPU support
with tf.name_scope('model'):
    # we will use these for the introspection of the kernels
    convWeights = []

    # add convolutional layer
    with tf.name_scope('convolution_layer_1'):
        [first, weights, biases] = buildConvLayer(X, 5, convStride, 2, poolStride, X.shape[3].value, 32)
        convWeights.append(weights)
    with tf.name_scope('convolution_layer_2'):
        [second, weights, biases] = buildConvLayer(first, 5, convStride, 2, poolStride, 32, 64)
        convWeights.append(weights)
    with tf.name_scope('convolution_layer_3'):
        [lastConvLayer, weights, biases] = buildConvLayer(second, 5, convStride, 2, poolStride, 64, 128)
        convWeights.append(weights)

    # project from convolution to neurons
    with tf.name_scope('projection'):
        reconnection = tf.reshape(lastConvLayer, [-1,4*4*128])

    # add two fully connected layers
    with tf.name_scope('fully_connected_layer_1'):
        fcl1 = buildFullyConnectedLayer(reconnection, 4*4*128, 2*1024, keep_prob)
    with tf.name_scope('fully_connected_layer_2'):
        fcl2 = buildFullyConnectedLayer(fcl1, 2*1024, 128, keep_prob)

    with tf.name_scope('out_layer'):
        # logits
        outweights = tf.Variable(0.01*tf.random_normal([128, numClasses]))
        out = tf.add(tf.matmul(fcl2, outweights), tf.Variable(0.01*tf.random_normal([numClasses])))

        # softmax
        pred = tf.nn.softmax(out)


# build optimizer
with tf.name_scope('optimization'):
    with tf.name_scope('cross_entropy'):
        crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=Y)
    with tf.name_scope('cost'):
        loss_op = tf.reduce_mean(crossentropy)
    with tf.name_scope('correct_predictions'):
        correct_prediction_op = tf.equal(tf.argmax(out, 1), tf.cast(Y, tf.int64))
    with tf.name_scope('accuracy'):
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction_op, tf.float64))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=global_step)

# initialize global variables
gvars = tf.global_variables_initializer()

# add loss and accuracy tracking
tf.summary.scalar('loss', loss_op, collections=['per_batch', 'per_epoch'])
tf.summary.scalar('accuracy', accuracy_op, collections=['per_batch', 'per_epoch'])

# if this is executed by a section, all previously defined summaries are evaluated
batch_summary_op = tf.summary.merge_all(key='per_batch')
epoch_summary_op = tf.summary.merge_all(key='per_epoch')

# ------- launch session -------
saver = tf.train.Saver()
with tf.Session() as sess:
    run_metadata = tf.RunMetadata()

    # init session
    sess.run(gvars)

    # restore last session
    checkpoint = tf.train.latest_checkpoint(logdir)
    if checkpoint:
        saver.restore(sess, checkpoint)
        print('Restored last session.')

    # save graph
    graph_writer = tf.summary.FileWriter(logdir+'/graph', sess.graph)
    graph_writer.close()

    print("Generating test and training sets...")
    # split data set according to 70-20-10 rule
    numSamples = len(indices)
    splitPoints = [int(numSamples*7/10), int(numSamples*9/10)]
    [trainingSetIndices, testSetIndices, validationSetIndices] = np.split(indices, splitPoints)
    [trainingSetLabels, testSetLabels, validationSetLabels] = np.split(labels[indices], splitPoints)

    trainingSet = images[trainingSetIndices]
    testSet = images[testSetIndices]
    validationSet = images[validationSetIndices]

    # build batches
    [batchesX, batchesY] = batchify(trainingSet, trainingSetLabels, batchSize)

    # some file writers
    hash = time.strftime("%Y%m%d%H%M%S")
    train_writer = tf.summary.FileWriter(logdir+'/train'+hash)
    validation_writer = tf.summary.FileWriter(logdir+'/validation'+hash)

    # optimize network
    print("Start training with %i epochs, starting from step %i..." % (numEpochs, global_step.eval()))
    learningStartTime = time.perf_counter()
    for epoch in range(numEpochs):
        for iter in range(len(batchesX)):
            index = global_step.eval()

            #print("Batch %d/%d in epoch %d/%d" % (iter+1, len(batchesX), epoch+1, numEpochs))
            # execute optimization step
            _, summary = sess.run([train_op, batch_summary_op], feed_dict={X: batchesX[iter], Y: batchesY[iter], keep_prob:pdropout})

            # save back statistics
            train_writer.add_summary(summary, index)

            # every few steps check performance against validation data set
            if (index % validationFreqency) == 0:
                summary = sess.run(epoch_summary_op, feed_dict={X: validationSet[:100], Y: validationSetLabels[:100], keep_prob:1.0})
                validation_writer.add_summary(summary, index)

                # approximate how much time is left
                timeUntilNow = time.perf_counter() - learningStartTime
                totalRuns = numEpochs*len(batchesX)
                currentRuns = epoch*len(batchesX) + iter + 1
                leftRuns = totalRuns - currentRuns
                timeLeft = leftRuns*timeUntilNow/currentRuns
                print('Done %d/%d. Approx time left %s ' % (currentRuns, totalRuns, pretty_time_delta(timeLeft)))
                #lastValidationStartTime = time.perf_counter()

        saver.save(sess, logdir, global_step=global_step)

    print("Optimization done!")
    validation_writer.close()
    train_writer.close()

    # introspect
    if introspect:
        introspection_writer = tf.summary.FileWriter(logdir+'/introspection'+hash)
        print('Introspecting network')
        # each convolution mask has the dimensionality [width, height, #channels in, #channels out]
        # and we want [#channels in, width, height, 1], as we want to visualize the individual
        # convolution masks per output channel
        with tf.name_scope('weights'):
            with tf.name_scope('convolutions'):
                for i, filtersInLayer in enumerate(convWeights):
                    for target in range(filtersInLayer.shape[2]):
                        # select [width, height, #channels out]
                        # and move the dimensions to the correct position
                        filterstack = tf.transpose(filtersInLayer[:,:,target,:], perm=[2,0,1])
                        # present the filters as grayscale images by adding
                        # a fourth dimension with a single index
                        filterstack = tf.reshape(filterstack, tf.expand_dims(filterstack, 3).shape)

                        tf.summary.image('source_'+str(target), filterstack, max_outputs=25, family='layer_'+str(i), collections=['introspection'])
                        print('Layer %i: %i/%i' % (i+1, target+1, filtersInLayer.shape[2].value))

        introspection_summary_op = tf.summary.merge_all(key='introspection')
        summary = sess.run(introspection_summary_op)
        introspection_writer.add_summary(summary)
        introspection_writer.close()

    # check performance
    [batchesX, batchesY] = batchify(testSet, testSetLabels, batchSize)
    totalAcc = 0.0
    for iter in range(len(batchesX)):
        [accuracy] = sess.run([accuracy_op], feed_dict={X: batchesX[iter], Y: batchesY[iter], keep_prob: 1.0})
        totalAcc += accuracy
    totalAcc /= len(batchesX)
    print("Test accuracy: %s" % (totalAcc))
