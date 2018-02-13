#TODO SPLIT SETS PROPERLY

from load_gtsrb import load_gtsrb_images
from util import pretty_time_delta, batchify, removeDiagonal

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import os
import time
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
class_descs = np.array(class_descs)
images = np.array(images)

#experiment with other color spaces
#images = np.array([ cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2YUV) for img in images])
#for i in range(len(images)):
#    images[i][:,:,0] = cv.equalizeHist(images[i][:,:,0])
#around 98% accuracy top performance, so slightly worse than raw RGB

# generate indices here, so they directly depend on the random seed an are
# therefore reproducible trough multiple runs, such that the sets are
# deterministically selected
indices = np.arange(len(images))
np.random.shuffle(indices)

print("Generating test and training sets...")
# split data set according to 70-20-10 rule
trainingSetIndices = np.array([], np.int64)
testSetIndices = np.array([], np.int64)
validationSetIndices = np.array([], np.int64)
trainingSetLabels = np.array([], np.uint32)
testSetLabels = np.array([], np.uint32)
validationSetLabels = np.array([], np.uint32)
# keep label distribution balanced
for currentLabel in range(numClasses):
    labelIndices = np.where(labels==currentLabel)[0]
    #np.random.shuffle(labelIndices)
    numSamples = len(labelIndices)
    splitPoints = [int(numSamples*7/10), int(numSamples*9/10)]

    [curTrainingSetIndices, curTestSetIndices, curValidationSetIndices] = np.split(labelIndices, splitPoints)
    trainingSetIndices = np.append(trainingSetIndices, curTrainingSetIndices)
    testSetIndices = np.append(testSetIndices, curTestSetIndices)
    validationSetIndices = np.append(validationSetIndices, curValidationSetIndices)

    [curTrainingSetLabels, curTestSetLabels, curValidationSetLabels] = np.split(labels[labelIndices], splitPoints)
    trainingSetLabels = np.append(trainingSetLabels, curTrainingSetLabels)
    testSetLabels = np.append(testSetLabels, curTestSetLabels)
    validationSetLabels = np.append(validationSetLabels, curValidationSetLabels)

#shuffle the training set, such that each batches has data with different labels
shuffleIndices = np.arange(len(trainingSetIndices))
np.random.shuffle(shuffleIndices)
trainingSet = images[trainingSetIndices[shuffleIndices]]
trainingSetLabels = trainingSetLabels[shuffleIndices]

testSet = images[testSetIndices]
validationSet = images[validationSetIndices]

# remember the global step, used for the tensorboard timeseries.
# this helps when the learning session gets somehow interrupted (e.g. hitting
# and OOM error or user interruption), we can smoothly continue the learning
global_step = tf.Variable(0, name='global_step', trainable=False)

# graph input
with tf.variable_scope('input'):
    # the 32x32 RGB input image stack
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    # the class label
    Y = tf.placeholder(tf.int32, [None])
    # probability to keep a unit when applying the dropout
    keep_prob = tf.placeholder(tf.float32)


def buildConvLayer(input, convSize, convStride, poolSize, poolStride, inDepth, outDepth, pdropout):
    '''
    Add a convolutional layer (similar to AlexNet) to a given input

    :param tensorflow.Tensor input: A tensor from which this layer recieves its input data
    :param int convSize: Size for the convolution mask
    :param [int int int int] convStride: Stride for the convolution mask
    :param int poolSize: Size of the pooling mask
    :param [int int int int] poolStride: Stride for the pooling mask
    :param int inDepth: Depth of the input layer
    :param int outDepth: Depth of this layer, which is equivalent to the input depth for the next layer
    :return [tensorflow.Tensor, tensorflow.Tensor, tensorflow.Tensor]:
    '''
    # generate variables
    weights = tf.Variable(0.01*tf.random_normal([convSize,convSize,inDepth, outDepth]))
    biases = tf.Variable(0.01*tf.random_normal([outDepth]))

    # add actual convolution
    conv = tf.nn.conv2d(input, weights, strides=convStride, padding='SAME')

    conv = tf.nn.dropout(conv, pdropout)

    # add an bias to the
    biased = tf.nn.bias_add(conv, biases)
    # connect the biased convolutions to the rectified linear unit
    relu = tf.nn.relu(biased)
    # pool the relus output
    pooling = tf.nn.max_pool(relu, ksize=[1,poolSize,poolSize,1], strides=poolStride, padding='SAME')

    # layer summarization for tensorboard
    tf.summary.histogram('weight_distribution', weights, collections=['per_epoch'])
    tf.summary.histogram('bias_distribution', biases, collections=['per_epoch'])
    tf.summary.histogram('relu_activation', relu, collections=['per_epoch'])

    return [pooling, weights, biases]


def buildFullyConnectedLayer(input, prevLayerSize, layerSize, pdropout):
    '''
    Add a convolutional layer (similar to the DCNN paper from 2012) to a given input

    :param tensorflow.Tensor input: A tensor from which this layer recieves its input data
    :param int prevLayerSize: Number of neurons in the previous layer
    :param int layerSize: Number of neurons in this layer
    :param float pdropout: Probability to keep a neuron in training
    :return [tensorflow.Tensor, tensorflow.Tensor, tensorflow.Tensor]:
    '''

    # build mapping
    weights = tf.Variable(0.01*tf.random_normal([prevLayerSize,layerSize]))
    biases = 0.01*tf.random_normal([layerSize])

    # MAC the input, weights and biases
    biased = tf.add(tf.matmul(input, weights), biases)
    # apply rectified linear unit
    relu = tf.nn.relu(biased)
    # end with a dropout
    dropout = tf.nn.dropout(relu, pdropout)

    # layer summarization for tensorboard
    tf.summary.histogram("weight_distribution", weights, collections=['per_epoch'])
    tf.summary.histogram("bias_distribution", biases, collections=['per_epoch'])
    tf.summary.histogram('relu_activation', relu, collections=['per_epoch'])

    return dropout

def plotSamplesWithRank(sampleSet, predictions, topX=5):
    for i, prediction in enumerate(predictions):
        # select the fife highest prediction probabilities
        indices = np.argsort(prediction)[::-1][:topX]

        # present the image anlong with the predicted probabilities and their labels
        ax = plt.subplot(1,len(predictions),i+1)
        plt.imshow(sampleSet[i].astype(np.uint8))
        labelText = ""
        for (conficence, labelName) in list(zip(prediction[indices], class_descs[indices])):
            labelText += '%f2.5 - %s\n' % (conficence, labelName)
        ax.set_xlabel(labelText)


def plotConfusionMatrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='hot'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    plt.ylabel('True label')
    plt.xlabel('Predicted label')





# build network with GPU support
with tf.name_scope('model'):
    # we will use these for the introspection of the kernels
    convWeights = []

    # add convolutional layer
    with tf.name_scope('convolution_layer_1'):
        [firstConvLayer, weights, biases] = buildConvLayer(X, 5, convStride, 2, poolStride, X.shape[3].value, 32, keep_prob)
        convWeights.append(weights)
    with tf.name_scope('convolution_layer_2'):
        [secondConvLayer, weights, biases] = buildConvLayer(firstConvLayer, 5, convStride, 2, poolStride, 32, 64, keep_prob)
        convWeights.append(weights)
    with tf.name_scope('convolution_layer_3'):
        [lastConvLayer, weights, biases] = buildConvLayer(secondConvLayer, 5, convStride, 2, poolStride, 64, 128, keep_prob)
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

#    numSamples = len(indices)
#    splitPoints = [int(numSamples*7/10), int(numSamples*9/10)]
#    [trainingSetIndices, testSetIndices, validationSetIndices] = np.split(indices, splitPoints)
#    [trainingSetLabels, testSetLabels, validationSetLabels] = np.split(labels[indices], splitPoints)

    # build batches
    batches = batchify(trainingSet, trainingSetLabels, batchSize)

    # some file writers
    hash = time.strftime("%Y%m%d%H%M%S")
    train_writer = tf.summary.FileWriter(logdir+'/train'+hash)
    validation_writer = tf.summary.FileWriter(logdir+'/validation'+hash)

    # optimize network
    print("Start training with %i epochs, starting from step %i..." % (numEpochs, global_step.eval()))
    learningStartTime = time.perf_counter()
    for epoch in range(numEpochs):
        for iter, batch in enumerate(batches):
            index = global_step.eval()

            # execute optimization step
            _, summary = sess.run([train_op, batch_summary_op], feed_dict={X: batch[0], Y: batch[1], keep_prob:pdropout})

            # save back statistics
            train_writer.add_summary(summary, index)

            # every few steps check performance against validation data set
            if (index % validationFreqency) == 0:
                #TODO use all validation data available
                summary = sess.run(epoch_summary_op, feed_dict={X: validationSet[:100], Y: validationSetLabels[:100], keep_prob:1.0})
                validation_writer.add_summary(summary, index)

                # approximate how much time is left
                timeUntilNow = time.perf_counter() - learningStartTime
                totalRuns = numEpochs*len(batches)
                currentRuns = epoch*len(batches) + iter + 1
                leftRuns = totalRuns - currentRuns
                timeLeft = leftRuns*timeUntilNow/currentRuns
                print('Done %d/%d. Approx time left %s ' % (currentRuns, totalRuns, pretty_time_delta(timeLeft)))

        saver.save(sess, logdir, global_step=global_step)

    print("Optimization done! Took %s" % pretty_time_delta(time.perf_counter() - learningStartTime))
    validation_writer.close()
    train_writer.close()

    # introspect
    if introspect:
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

        # save introspection back to file
        introspection_writer = tf.summary.FileWriter(logdir+'/introspection'+hash)
        introspection_summary_op = tf.summary.merge_all(key='introspection')
        summary = sess.run(introspection_summary_op)
        introspection_writer.add_summary(summary)
        introspection_writer.close()

    # check performance
    print("Evaluating prediction performance.")
    batches = batchify(testSet, testSetLabels, batchSize)
    # gather accuracy
    totalAcc = 0.0
    # and information about the predicted labels
    allPredictionIndices = np.array([])
    topPredictionLabels = np.empty((len(testSet), numClasses))
    for i, batch in enumerate(batches):
        [accuracy, predictionIndices, predictionValues] = sess.run([accuracy_op, correct_prediction_op, pred], feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})
        allPredictionIndices = np.append(allPredictionIndices, predictionIndices)
        for j in range(predictionValues.shape[0]):
            topPredictionLabels[i*batchSize+j, :] = np.argsort(predictionValues[j,:])[::-1]
        totalAcc += accuracy
    totalAcc /= len(batches)

    print('Test accuracy: %s' % (totalAcc))

    # save back meta information
    cnfmat = sess.run(tf.contrib.metrics.confusion_matrix(topPredictionLabels[:, 0], testSetLabels))
    plotConfusionMatrix(removeDiagonal(cnfmat), classes=class_descs)
    plt.show()

    topLabelsHistogram = np.zeros((numClasses, numClasses))
    for i in range(len(testSet)):
        for j in range(numClasses):
            predictedLabel = int(topPredictionLabels[i, j])
            trueLabel = int(testSetLabels[i])
            topLabelsHistogram[predictedLabel, j] += predictedLabel == trueLabel

    plt.imshow(np.log(topLabelsHistogram+1), cmap='hot')
    plt.title('Log-Scaled Histogram')
    plt.colorbar()
    plt.xlabel('Rank')
    plt.ylabel('Label')
    plt.show()


    # show examples
    print('Present some examples....')

    # start by selecting random examples from correctly classified
    exampleIndices = np.where(allPredictionIndices)[0]
    np.random.shuffle(exampleIndices)
    exampleIndices = exampleIndices[:3]

    [predictions] = sess.run([pred], feed_dict={X:testSet[exampleIndices], Y:trainingSetLabels[exampleIndices], keep_prob:1.0})

    plotSamplesWithRank(testSet[exampleIndices], predictions)
    plt.suptitle('Examples for correct predictions')
    plt.show()

    # now show some examples where the classification failed
    exampleIndices = np.where(allPredictionIndices == False)[0]
    np.random.shuffle(exampleIndices)
    exampleIndices = exampleIndices[:3]

    [predictions] = sess.run([pred], feed_dict={X:testSet[exampleIndices], Y:trainingSetLabels[exampleIndices], keep_prob:1.0})

    plotSamplesWithRank(testSet[exampleIndices], predictions)
    plt.suptitle('Examples for bad predictions')
    plt.show()

    """
    for goodImage in testSet[np.where(allPredictionIndices == True)[0]][:5]:
        plt.imshow(goodImage.astype(np.uint8))
        plt.show()

        [layer1Response] = sess.run([firstConvLayer], feed_dict={X: np.reshape(np.array([goodImage]),[1,32,32,3]), keep_prob:1.0})

        outerSize = int(np.ceil(np.sqrt(layer1Response.shape[3])))
        for j in range(layer1Response.shape[0]):
            for i in range(layer1Response.shape[3]):
                plt.subplot(outerSize, outerSize, i+1)
                plt.imshow(layer1Response[0,:,:,i].astype(np.uint8))
            plt.show()

        [layer2Response] = sess.run([secondConvLayer], feed_dict={X: np.reshape(np.array([goodImage]),[1,32,32,3]), keep_prob:1.0})

        outerSize = int(np.ceil(np.sqrt(layer2Response.shape[3])))
        for j in range(layer2Response.shape[0]):
            for i in range(layer2Response.shape[3]):
                plt.subplot(outerSize, outerSize, i+1)
                plt.imshow(layer2Response[j,:,:,i].astype(np.uint8))
            plt.show()

        [layer3Response] = sess.run([lastConvLayer], feed_dict={X: np.reshape(np.array([goodImage]),[1,32,32,3]), keep_prob:1.0})

        outerSize = int(np.ceil(np.sqrt(layer3Response.shape[3])))
        for j in range(layer3Response.shape[0]):
            for i in range(layer3Response.shape[3]):
                plt.subplot(outerSize, outerSize, i+1)
                plt.imshow(layer3Response[j,:,:,i].astype(np.uint8))
            plt.show()
    """

    # show some failed images with their layerwise responses
    for failImage in testSet[np.where(allPredictionIndices == False)[0]][:5]:
        #plt.imshow(cv.cvtColor(failImage.astype(np.uint8), cv.COLOR_YUV2RGB))
        plt.imshow(failImage.astype(np.uint8))
        plt.show()
"""
        [layer1Response] = sess.run([firstConvLayer], feed_dict={X: np.reshape(np.array([failImage]),[1,32,32,3]), keep_prob:1.0})

        outerSize = int(np.ceil(np.sqrt(layer1Response.shape[3])))
        for j in range(layer1Response.shape[0]):
            for i in range(layer1Response.shape[3]):
                plt.subplot(outerSize, outerSize, i+1)
                plt.imshow(layer1Response[0,:,:,i].astype(np.uint8))
            plt.show()

        [layer2Response] = sess.run([secondConvLayer], feed_dict={X: np.reshape(np.array([failImage]),[1,32,32,3]), keep_prob:1.0})

        outerSize = int(np.ceil(np.sqrt(layer2Response.shape[3])))
        for j in range(layer2Response.shape[0]):
            for i in range(layer2Response.shape[3]):
                plt.subplot(outerSize, outerSize, i+1)
                plt.imshow(layer2Response[j,:,:,i].astype(np.uint8))
            plt.show()

        [layer3Response] = sess.run([lastConvLayer], feed_dict={X: np.reshape(np.array([failImage]),[1,32,32,3]), keep_prob:1.0})

        outerSize = int(np.ceil(np.sqrt(layer3Response.shape[3])))
        for j in range(layer3Response.shape[0]):
            for i in range(layer3Response.shape[3]):
                plt.subplot(outerSize, outerSize, i+1)
                plt.imshow(layer3Response[j,:,:,i].astype(np.uint8))
            plt.show()
"""
