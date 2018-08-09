"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import math
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Iterator
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.framework.errors_impl import OutOfRangeError

from alexnet import AlexNet, SiameseAlexNet
from datagenerator import ImageDataGenerator
from checkpoint import Checkpointer, MemCache
from metrics import Metrics
from utils import auto_adapt_batch, determine_list, print_info, get_environment_parameters, get_init_op

"""
Configuration Part.
"""

np.set_printoptions(threshold=np.inf)

# define a function to write metrics_dict for floydhub to parse
sMetrics = Metrics(beta=0.)
aMetrics = Metrics(beta=0.)

opt = get_environment_parameters()
print(opt)

# Learning params
aLR, sLR = opt.lr1, opt.lr2
aEpochs, sEpochs = opt.nepochs1, opt.nepochs2

# Network params
dropout_rate = opt.dropout
if opt.nclasses == 0:
    if opt.val2 and opt.train2:
        num_classes = 3
    else:
        num_classes = 2
else:
    num_classes = opt.nclasses
print('There are %d labels for classification' % num_classes)
# train_layers = opt.trainLayers.split()
aTrainLayers, sTrainLayers = set(opt.trainLayers1.split()), set(opt.trainLayers2.split())
# train_layers = aTrainLayers | sTrainLayers
# How often we want to write the tf.summary data to disk
display_step = opt.displayStep
assert opt.checkStd in ['acc', 'xent'], 'Illegal check standard, %s' % opt.checkStd

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = opt.outf
checkpoint_path = os.path.join(opt.outf, 'checkpoints')
sample_path = os.path.join(opt.outf, 'samplelist')

train_file, train_length, val_file, val_length, test_file, test_length = determine_list(opt, sample_path)

batch_size1 = opt.batchSize1 if opt.batchSize1 else auto_adapt_batch(train_length, val_length)
batch_size2 = opt.batchSize2 if opt.batchSize2 else auto_adapt_batch(train_length, val_length) // 2

# TODO overall debugging, paying special attention to the instantiation of tf.Session
"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    train_data = ImageDataGenerator(train_file,
                                    mode='training',
                                    batch_size=batch_size1,
                                    num_classes=num_classes,
                                    shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size1,
                                  num_classes=num_classes,
                                  shuffle=True)
    test_data = ImageDataGenerator(test_file,
                                   mode='inference',
                                   batch_size=batch_size1,
                                   num_classes=num_classes,
                                   shuffle=True)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(train_data.data.output_types,
                                       train_data.data.output_shapes)
    next_batch = iterator.get_next()
print('data loaded and preprocessed on the cpu')

# Ops for initializing the two different iterators
training_init_op = get_init_op(iterator, train_data)
validation_init_op = get_init_op(iterator, val_data)
testing_init_op = get_init_op(iterator, test_data)

# TF placeholder for graph input and output
# y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize the FileWriter
train_writer = tf.summary.FileWriter(os.path.join(filewriter_path, 'train'))
val_writer = tf.summary.FileWriter(os.path.join(filewriter_path, 'val'))

# Initialize an saver for store model checkpoints
# saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
aTrainBatches = int(train_data.data_size // batch_size1)
aValBatches = int(val_data.data_size // batch_size1)
aTestBatches = int(test_data.data_size // batch_size1)

sTrainBatches = int(train_data.data_size // batch_size2) // 2
sValBatches = int(val_data.data_size // batch_size2) // 2
sTestBatches = int(test_data.data_size // batch_size2) // 2

print_info(batch_size1, aTrainBatches, aValBatches, aTestBatches, train_length, val_length, test_length,
           model='AlexNet')
print_info(batch_size2, sTrainBatches * 2, sValBatches * 2, sTestBatches * 2, train_length, val_length, test_length,
           model='SiameseAlexNet')

# create and prepare a Siamese AlexNet
# create 2 placeholders for the AlexNets nested in Siamese Net
x1 = tf.placeholder(tf.float32, [None, 227, 227, 3], name='x1')
x2 = tf.placeholder(tf.float32, [None, 227, 227, 3], name='x2')
# Construct the Siamese Model
sNet = SiameseAlexNet(x1, x2, keep_prob, num_classes, sTrainLayers, weights_path=opt.pretrained,
                      margin00=opt.margin00, margin01=opt.margin01, margin11=opt.margin11,
                      punish00=opt.punish00, punish01=opt.punish01, punish11=opt.punish11,
                      proj=opt.siamese)
y1, y2 = sNet.net1.y, sNet.net2.y  # get the label placeholders of the Siamese Net
sVars = [v for v in tf.trainable_variables() if v.name.split('/')[-2] in sTrainLayers]
sLoss = sNet.loss  # get loss tensor of the Siamese Net
with tf.name_scope("siamese-train"):  # define the train_op of Siamese Training
    sGradients = tf.gradients(sLoss, sVars)
    sGradients = list(zip(sGradients, sVars))

    # get rid of the null elements
    sGradients = [g_and_v for g_and_v in sGradients if g_and_v[0] is not None]
    sVars = [g_and_v[1] for g_and_v in sGradients]
    print('Siamese Variables are:', sVars)

    sOptimizer = tf.train.GradientDescentOptimizer(sLR)
    sTrainOp = sOptimizer.apply_gradients(grads_and_vars=sGradients)
for gradient, var in sGradients:  # write summary of gradients
    tf.summary.histogram(var.name + "/gradient-Siamese", gradient)
for var in sVars:  # write summary of variables
    tf.summary.histogram(var.name, var)
sLoss_summ = tf.summary.scalar('siamese-loss', sLoss)  # write summary of siamese loss

# Siamese Training for n1 epochs
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer.add_graph(sess.graph)
    sNet.load_model_pretrained(sess)
    # initialize a best-metrics update mechanism, i.e. checkpointer
    ckpt_path = os.path.join(checkpoint_path, "siam.npy")
    checkpointer = Checkpointer("Siamese Loss(Val)", sNet, ckpt_path, higher_is_better=False, sess=sess)

    # bach sizes are divided by half in siamese training
    # dealt with NaN in siamese training
    # REVIEW Val-Loss is way larger than Train-Loss, possibly due to different dropout rate in train and val
    for epoch in range(sEpochs):
        print("------- Siamese Epoch number: {} ------- ".format(epoch + 1))

        print("start training, %d batches in total" % sTrainBatches)
        sess.run(training_init_op)
        for step in range(sTrainBatches):
            try:
                img_batch1, label_batch1 = sess.run(next_batch)
                img_batch2, label_batch2 = sess.run(next_batch)
                sess.run([sTrainOp], feed_dict={x1: img_batch1,
                                                x2: img_batch2,
                                                y1: label_batch1,
                                                y2: label_batch2,
                                                keep_prob: 1.})
                s, _loss = sess.run([sLoss_summ, sLoss], feed_dict={x1: img_batch1,
                                                                    x2: img_batch2,
                                                                    y1: label_batch1,
                                                                    y2: label_batch2,
                                                                    keep_prob: 1.})
                print("step: %d, loss = %f" % (step, _loss))
                count00, count11, count01, loss00, loss11, loss01 = \
                    sess.run([sNet.count00, sNet.count11, sNet.count01, sNet.loss00, sNet.loss11, sNet.loss01],
                             feed_dict={x1: img_batch1,
                                        x2: img_batch2,
                                        y1: label_batch1,
                                        y2: label_batch2,
                                        keep_prob: 1.})
                print('neg-neg-count =', count00, 'mean-neg-neg-loss =', loss00)
                print('pos-pos-count =', count11, 'mean-pos-pos-loss =', loss11)
                print('neg-pos-count =', count01, 'mean-neg-pos-loss =', loss01)
                if (epoch * sTrainBatches + step) % display_step == 0:
                    train_writer.add_summary(s)
            except OutOfRangeError as e:
                print(e)
                print('ignoring residue batches in step %d' % step)
            except Exception as e:
                print(e)
                print('some other exception occurred in step %d' % step)
        print("start validation, %d batches in total" % sValBatches)
        sess.run(validation_init_op)
        val_loss = 0
        for step in range(sValBatches):
            try:
                img_batch1, label_batch1 = sess.run(next_batch)
                img_batch2, label_batch2 = sess.run(next_batch)
                count00, count11, count01, loss00, loss11, loss01 = \
                    sess.run([sNet.count00, sNet.count11, sNet.count01, sNet.loss00, sNet.loss11, sNet.loss01],
                             feed_dict={x1: img_batch1,
                                        x2: img_batch2,
                                        y1: label_batch1,
                                        y2: label_batch2,
                                        keep_prob: 1.})
                print('neg-neg-count =', count00, 'mean-neg-neg-loss =', loss00)
                print('pos-pos-count =', count11, 'mean-pos-pos-loss =', loss11)
                print('neg-pos-count =', count01, 'mean-neg-pos-loss =', loss01)
                s, step_loss = sess.run([sLoss_summ, sLoss], feed_dict={x1: img_batch1,
                                                                        x2: img_batch2,
                                                                        y1: label_batch1,
                                                                        y2: label_batch2,
                                                                        keep_prob: 1.})
                print("step: %d, loss = %f" % (step, step_loss))
                val_loss += step_loss
                if (epoch * sValBatches + step) % display_step == 0:
                    val_writer.add_summary(s)
            except OutOfRangeError as e:
                print(e)
                print('ignoring residue batches in step %d' % step)
            except Exception as e:
                print(e)
                print('some other exception occurred in step %d' % step)
        val_loss /= sValBatches
        sMetrics.update_metric('siam-loss-val', val_loss)
        sMetrics.write_metrics()

        # do checkpointing
        checkpointer.update_best(val_loss, checkpoint=(not opt.noCheck), mem_cache=True, epoch=epoch)

        # re-shuffle the training set to generate new pairs for siamese training
        train_data.reshuffle_data()

    # after training, save the parameters corresponding to the lowest losses
    mem_caches = checkpointer.list_memory_caches()

# get and prepare the AlexNet of that Siamese Net
# set the parameters to `mem_size` sets of parameters with lowest losses, currently using only the first one
# with tf.Session() as sess:
#     sNet.set_model_vars(mem_caches[0].get_parameters(), sess)

x = tf.placeholder(tf.float32, [None, 227, 227, 3], name="x")  # get the input placeholder
with tf.Session() as sess:  # double-checked that train_layers are what they are supposed to be
    sess.run(tf.global_variables_initializer())
    aNet = sNet.get_net_copy(sess, x=x, train_layers=aTrainLayers)  # get an AlexNet with trained Variables
# aNet = AlexNet(x, keep_prob, num_classes, aTrainLayers, weights_path=opt.pretrained)
y = aNet.y  # grab the label placeholder of the alexNet
aVars = [v for v in tf.trainable_variables() if v.name.split('/')[-2] in aTrainLayers]
# get the metrics of the AlexNet
aLoss, accuracy, precision, recall, F_alpha = aNet.loss, aNet.accuracy, aNet.precision, aNet.recall, aNet.F_alpha
with tf.name_scope("classification-train"):
    aGradients = tf.gradients(aLoss, aVars)
    aGradients = list(zip(aGradients, aVars))

    # get rid of the null elements
    aGradients = [g_and_v for g_and_v in aGradients if g_and_v[0] is not None]
    aVars = [g_and_v[1] for g_and_v in aGradients]
    print('AlexNet Variables are:', aVars)

    aOptimizer = tf.train.GradientDescentOptimizer(aLR)
    aTrainOp = aOptimizer.apply_gradients(grads_and_vars=aGradients)
for gradient, var in aGradients:
    tf.summary.histogram(var.name + "/gradients-AlexNet", gradient)
for var in aVars:
    tf.summary.histogram(var.name, var)
# get the metric summaries of the AlexNet
aLoss_summ = tf.summary.scalar('xent-loss', aLoss)
accuracy_summ = tf.summary.scalar('accuracy', accuracy)
precision_summ = tf.summary.scalar('precision', precision)
recall_summ = tf.summary.scalar('recall', recall)
F_alpha_summ = tf.summary.scalar('F_alpha', F_alpha)
alexnet_summ = tf.summary.merge([aLoss_summ, accuracy_summ, precision_summ, recall_summ, F_alpha_summ])


# Classification Training for n2 epochs
def alexnet_training(aNet, param_set_id):
    global sess, ckpt_path, checkpointer, epoch, step, s, val_loss, step_loss
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)
        # since the aNet is obtained from sNet, pre-trained weights does not need to be loaded

        ckpt_path = os.path.join(checkpoint_path, "alex.npy")
        checkpointer = Checkpointer(opt.checkStd, aNet, ckpt_path, higher_is_better=(opt.checkStd == "acc"), sess=sess)

        for epoch in range(aEpochs):
            print("------- Parameter Set: {}, AlexNet Epoch number: {} ------- ".format(param_set_id, epoch + 1))

            print("start training, %d batches in total" % aTrainBatches)
            sess.run(training_init_op)
            for step in range(aTrainBatches):
                img_batch, label_batch = sess.run(next_batch)
                sess.run(aTrainOp, feed_dict={x: img_batch, y: label_batch, keep_prob: 1. - dropout_rate})
                # TP, TN, FP, FN = sess.run([aNet.TP, aNet.TN, aNet.FP, aNet.FN],
                #                           feed_dict={x: img_batch, y: label_batch, keep_prob: 1. - dropout_rate})
                # print("TP = %d, TN = %d, FP = %d, FN = %d" % (TP, TN, FP, FN))
                if (epoch * aTrainBatches + step) % display_step == 0:
                    # print("TP = %d, TN = %d, FP = %d, FN = %d" % (TP, TN, FP, FN))
                    s = sess.run(alexnet_summ, feed_dict={x: img_batch, y: label_batch, keep_prob: 1 - dropout_rate})
                    train_writer.add_summary(s)

            print("start validation, %d batches in total" % aValBatches)
            sess.run(validation_init_op)
            val_loss, val_acc, val_precision, val_recall, val_F_alpha = 0., 0., 0., 0., 0.
            for step in range(aValBatches):
                img_batch, label_batch = sess.run(next_batch)
                TP, TN, FP, FN = sess.run([aNet.TP, aNet.TN, aNet.FP, aNet.FN], feed_dict={x: img_batch,
                                                                                           y: label_batch,
                                                                                           keep_prob: 1.})
                print("TP = %d, TN = %d, FP = %d, FN = %d" % (TP, TN, FP, FN))
                s, step_loss, step_acc, step_precision, step_recall, step_F_alpha = sess.run(
                    [alexnet_summ, aLoss, accuracy, precision, recall, F_alpha],
                    feed_dict={x: img_batch, y: label_batch, keep_prob: 1.}
                )
                print("precision = {}, recall = {}, F_alpha = {}".format(step_precision, step_recall, step_F_alpha))
                val_loss += step_loss
                val_acc += step_acc
                val_precision += step_precision
                val_recall += step_recall
                val_F_alpha += step_F_alpha
                if (epoch * aValBatches + step) % display_step == 0:
                    val_writer.add_summary(s)
            val_loss /= aValBatches
            val_acc /= aValBatches
            val_precision /= aValBatches
            val_recall /= aValBatches
            val_F_alpha /= aValBatches
            metric_names = ['val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_F_alpha']
            aMetrics.update_metrics(metric_names, [val_loss, val_acc, val_precision, val_recall, val_F_alpha])
            aMetrics.write_metrics()

            # do checkpointing
            checkpointer.update_best(val_loss, checkpoint=(not opt.noCheck))

            # re-shuffle data to observe model behaviour
            train_data.reshuffle_data()


for index, mem_cache in enumerate(mem_caches):  # type: int, MemCache
    print("+" * 15, "proceeding with parameters set No. %d" % (index + 1), "+" * 15)
    with tf.Session() as sess:
        aNet.set_model_vars(mem_cache.get_parameters(), sess)

    alexnet_training(aNet, index)
