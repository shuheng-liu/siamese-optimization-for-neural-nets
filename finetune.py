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

import argparse
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
from checkpoint import Checkpointer
from metrics import Metrics

"""
Configuration Part.
"""

np.set_printoptions(threshold=np.inf)


# generate a txt file containing image paths and labels
def make_list(folders, flags=None, ceils=None, mode='train', store_path='/output'):
    suffices = ('jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG')
    if ceils is None: ceils = [-1] * len(folders)  # ceil constraint not imposed
    if flags is None: flags = list(range(len(folders)))  # flags = [0, 1, ..., n-1]
    assert len(folders) == len(flags) == len(ceils), (len(folders), len(flags), len(ceils))
    assert mode in ['train', 'val', 'test']
    folders_flags_ceils = [tup for tup in zip(folders, flags, ceils)
                           if isinstance(tup[0], str) and os.path.isdir(tup[0])]
    assert folders_flags_ceils

    print('Making %s list' % mode)
    for tup in folders_flags_ceils:
        print('Folder {}: flag = {}, ceil = {}'.format(*tup))
    if not os.path.isdir(store_path): os.mkdir(store_path)
    out_list = os.path.join(store_path, mode + '.txt')
    list_length = 0
    with open(out_list, 'w') as fo:
        for (folder, flag, ceil) in folders_flags_ceils:
            count = 0
            for pic_name in os.listdir(folder):
                if pic_name.split('.')[-1] not in suffices:
                    print('Ignoring non-image file {} in folder {}.'.format(pic_name, folder),
                          'Legal suffices are', suffices)
                    continue
                count += 1
                list_length += 1
                fo.write("{} {}\n".format(os.path.join(folder, pic_name), flag))
                # if ceil is imposed (ceil > 0) and count exceeds ceil, break and write next flag
                if 0 < ceil <= count: break
    print('%s list made\n' % mode)
    return out_list, list_length


# find a suitable batchSize
# TODO redefine auto_adatpt_batch() for suitable behaviour of Siamese Training
def auto_adapt_batch(train_size, val_size, batch_count_multiple=1, max_size=256):
    '''
    returns a suitable batch size according to train and val dataset size,
    say max_size = 128, and val_size is smaller than train_size,
        if val_size < 128, the batch_size1 to be returned is val_size
        if 128 < val_size <= 256, the batch size is 1/2 of val_size, at most 1 validation sample cannot be used
        if 256 < val_size <= 384, the batch size is 1/3 of val_size, at most 2 validation samples cannot be used
        ...
    :param train_size: the number of training samples in the training set
    :param val_size: the number of validation samples in the validation set
    :param max_size: the maximum batch_size1 that is allowed to be returned
    :param batch_count_multiple: force the batch count to be a multiple of this number, default = 1
    :return: a suitable batch_size1 for the input
    '''
    print('Auto adapting batch size...')
    numerator = min(train_size, val_size)
    denominator = 0
    while True:
        denominator += batch_count_multiple
        batch_size = numerator // denominator
        if batch_size <= max_size: return batch_size


# define a function to write metrics_dict for floydhub to parse
sMetrics = Metrics(beta=0.)
aMetrics = Metrics(beta=0.)


def get_environemnt_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train0', required=True, help='paths to negative training dataset, separated by space')
    parser.add_argument('--train1', required=True, help='paths to positive training dataset, separated by space')
    parser.add_argument('--train2', default='', help='paths to other disease training dataset, separated by space')
    parser.add_argument('--trainCeils', default=None, help='Ceils of Training')
    parser.add_argument('--val0', required=True, help='paths to negative validation dataset, separated by space')
    parser.add_argument('--val1', required=True, help='paths to positive validation dataset, separated by space')
    parser.add_argument('--val2', default='', help='paths to other disease validation dataset, separated by space')
    parser.add_argument('--valCeils', default=None, help='Ceils of validation')
    parser.add_argument('--lr1', type=float, default=1e-3, help='learning rate for supervised learning, default=1e-3')
    parser.add_argument('--lr2', type=float, default=5e-7, help='learning rate for siamese learning, default=5e-7')
    parser.add_argument('--nepochs1', type=int, default=100, help='number of supervised epochs, default = 100')
    parser.add_argument('--nepochs2', type=int, default=100, help='number of siamese epochs, default = 100')
    parser.add_argument('--batchSize1', type=int, default=0, help='default = automatic-adapting')
    parser.add_argument('--batchSize2', type=int, default=0, help='default = automatic-adapting')
    parser.add_argument('--dropout', type=int, default=0.5, help='dropout rate for alexnet, default = 0.5')
    parser.add_argument('--nclasses', type=int, default=2, help='number of classes, default = 2')
    parser.add_argument('--trainLayers1', type=str, default='fc7 fc8', help='default = fc7 fc8')
    parser.add_argument('--trainLayers2', type=str, default='fc6', help='default = fc6')
    parser.add_argument('--displayStep', type=int, default=20, help='How often to write tf.summary')
    parser.add_argument('--outf', type=str, default='/output', help='path for checkpoints & tf.summary & samplelist')
    parser.add_argument('--pretrained', type=str, default='/', help='path for pre-trained weights *.npy')
    parser.add_argument('--noCheck', action='store_true', help='don\'t save model checkpoints')
    parser.add_argument('--siamese', type=str, default='dropout6', help='siamese projection layers, default=dropout6')
    parser.add_argument('--checkStd', type=str, default='xent', help='Standard for checkpointing, acc or xent')
    parser.add_argument('--margin00', type=float, default=8.0, help='distance margin for neg-neg pair, default=10.0')
    parser.add_argument('--margin11', type=float, default=6.0, help='distance margin for pos-pos pair, default=8.0')
    parser.add_argument('--margin01', type=float, default=7.07, help='distance margin for neg-pos pair, default=7.0')
    # parser.add_argument('--margin00', type=float, default=8.0, help='distance margin for neg-neg pair, default=10.0')
    # parser.add_argument('--margin11', type=float, default=10.0, help='distance margin for pos-pos pair, default=8.0')
    # parser.add_argument('--margin01', type=float, default=4.0, help='distance margin for neg-pos pair, default=7.0')
    return parser.parse_args()


opt = get_environemnt_parameters()
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


# make train & val & test list, and do stats
def determine_list(opt):
    # paths
    train0, train1, train2 = opt.train0.split(), opt.train1.split(), opt.train2.split()
    val0, val1, val2 = opt.val0.split(), opt.val1.split(), opt.val2.split()
    train, val = train0 + train1 + train2, val0 + val1 + val2
    test = train + val
    # flags
    train_flags = [0] * len(train0) + [1] * len(train1) + [2] * len(train2)
    val_flags = [0] * len(val0) + [1] * len(val1) + [2] * len(val2)
    test_flags = train_flags + val_flags
    # ceils
    train_ceils = opt.trainCeils.split() if opt.trainCeils else [-1] * len(train_flags)
    train_ceils = [int(c) for c in train_ceils]
    val_ceils = opt.valCeils.split() if opt.valCeils else [-1] * len(train_flags)
    val_ceils = [int(c) for c in val_ceils]
    test_ceils = train_ceils + val_ceils
    # do list generating
    train_file, train_length = make_list(train, flags=train_flags, ceils=train_ceils, mode='train',
                                         store_path=sample_path)
    val_file, val_length = make_list(val, flags=val_flags, mode='val', ceils=val_ceils, store_path=sample_path)
    test_file, test_length = make_list(test, flags=test_flags, mode='test', ceils=test_ceils, store_path=sample_path)
    return train_file, train_length, val_file, val_length, test_file, test_length


train_file, train_length, val_file, val_length, test_file, test_length = determine_list(opt)

batch_size1 = opt.batchSize1 if opt.batchSize1 else auto_adapt_batch(train_length, val_length)
batch_size2 = opt.batchSize2 if opt.batchSize2 else auto_adapt_batch(train_length, val_length) // 2


# print the info about training, validation and testing sets
def print_info(batch_size, train_batches, va_batches, test_batches, model='AlexNet'):
    assert model in ['SiameseAlexNet', 'AlexNet']

    print("********** In model %s **********" % model)
    print('%d samples in training set' % train_length)
    print('%d samples in validation set' % val_length)
    print('Train-Val ratio == %.1f%s : %.1f%s' % (100 * train_length / (train_length + val_length), '%',
                                                  100 * val_length / (train_length + val_length), '%'))
    print('Batch Size =', batch_size)
    print('Of all %d val samples, %d is utilized, percentage = %.1f%s' % (val_length, va_batches * batch_size,
                                                                          va_batches * batch_size / val_length
                                                                          * 100, '%'))
    print('Of all %d train samples, %d is utilized, percentage = %.1f%s' % (train_length, train_batches * batch_size,
                                                                            train_batches * batch_size / train_length
                                                                            * 100, '%'))
    print('Of all %d test samples, %d is utilized, percentage = %.1f%s' % (test_length, test_batches * batch_size,
                                                                           test_batches * batch_size / test_length
                                                                           * 100, '%'))


# TODO overall debugging
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
def get_init_op(iterator, some_data: ImageDataGenerator):
    return iterator.make_initializer(some_data.data)


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

print_info(batch_size1, aTrainBatches, aValBatches, aTestBatches, model='AlexNet')
print_info(batch_size2, sTrainBatches * 2, sValBatches * 2, sTestBatches * 2, model='SiameseAlexNet')

# create and prepare a Siamese AlexNet
# create 2 placeholders for the AlexNets nested in Siamese Net
x1 = tf.placeholder(tf.float32, [None, 227, 227, 3], name='x1')
x2 = tf.placeholder(tf.float32, [None, 227, 227, 3], name='x2')
# Construct the Siamese Model
sNet = SiameseAlexNet(x1, x2, keep_prob, num_classes, sTrainLayers, weights_path=opt.pretrained,
                      margin00=opt.margin00, margin01=opt.margin01, margin11=opt.margin11, proj=opt.siamese)
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
    # TODO debug: Val-Loss is way larger than Train-Loss, possibly due to different dropout rate in train and val
    for epoch in range(sEpochs):
        print("------- Siamese Epoch number: {} ------- ".format(epoch + 1))

        print("start training, %d batches in total" % sTrainBatches)
        sess.run(training_init_op)
        for step in range(sTrainBatches):
            try:
                img_batch1, label_batch1 = sess.run(next_batch)
                img_batch2, label_batch2 = sess.run(next_batch)
                # TODO re-enable sTrainOp
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
# UNDONE set the parameters to `mem_size` sets of parameters with lowest losses, currently using only the first one
sNet.set_model_vars(mem_caches[0].get_parameters(), checkpointer.session)

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
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer.add_graph(sess.graph)
    # since the aNet is obtained from sNet, pre-trained weights does not need to be loaded

    ckpt_path = os.path.join(checkpoint_path, "alex.npy")
    checkpointer = Checkpointer(opt.checkStd, aNet, ckpt_path, higher_is_better=(opt.checkStd == "acc"), sess=sess)

    for epoch in range(aEpochs):
        print("------- AlexNet Epoch number: {} ------- ".format(epoch + 1))

        print("start training, %d batches in total" % aTrainBatches)
        sess.run(training_init_op)
        for step in range(aTrainBatches):
            img_batch, label_batch = sess.run(next_batch)
            sess.run(aTrainOp, feed_dict={x: img_batch, y: label_batch, keep_prob: 1. - dropout_rate})
            # TP, TN, FP, FN = sess.run([aNet.TP, aNet.TN, aNet.FP, aNet.FN], feed_dict={x: img_batch,
            #                                                                            y: label_batch,
            #                                                                            keep_prob: 1. - dropout_rate})
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
            val_loss += step_loss
            val_acc += step_acc
            val_precision += step_precision
            val_recall += step_recall
            val_F_alpha += step_F_alpha
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

# # Initialize model
# if opt.siamese:
#     model = AlexNet((x1, x2), keep_prob, num_classes, train_layers, weights_path=opt.pretrained, margin=opt.margin)
# else:
#     model = AlexNet(x1, keep_prob, num_classes, train_layers, weights_path=opt.pretrained)
#
# y1, y2 = model.y1, model.y2
# # Link variable to model output
# score = model.fc8
# projections = model.fc7  # i.e. embeddings, not to be mistaken with `embeddings` belows
#
# # List of trainable variables of the layers we want to train
# vars = [v for v in tf.trainable_variables() if v.name.split('/')[-2] in train_layers]
# # sVars = [v for v in tf.trainable_variables() if v.name.split('/')[-2] in sTrainLayers]
#
# xent_loss = model.loss
# if opt.siamese: siamese_loss = model.quadratic_siamese_loss
#
# # Train op
# with tf.name_scope("supervised-train"):
#     # Get gradients of all trainable variables
#     gradients = tf.gradients(xent_loss, aVars)
#     print('supervised gradients', gradients)
#     gradients = list(zip(gradients, aVars))
#     print('supervised gradients', gradients)
#
#     # Create optimizer and apply gradient descent to the trainable variables
#     optimizer = tf.train.GradientDescentOptimizer(aLR)
#     supervised_train_op = optimizer.apply_gradients(grads_and_vars=gradients)
#
# # Add gradients to summary
# for gradient, var in gradients:
#     tf.summary.histogram(var.name + '/gradient-supervised', gradient)
#
# # duplicate of the above process, defining siamese_train_op
# if opt.siamese:
#     with tf.name_scope("siamese-train"):
#         gradients = tf.gradients(siamese_loss, sVars)
#         print('siamese gradients', gradients)
#         gradients = list(zip(gradients, sVars))
#         print('siamese gradients', gradients)
#         optimizer = tf.train.GradientDescentOptimizer(sLR)
#         siamese_train_op = optimizer.apply_gradients(grads_and_vars=gradients)
#
#     for gradient, var in gradients:
#         tf.summary.histogram(var.name + '/gradient-siamese', gradient)
#
# # Add the variables we train to the summary
# for var in vars:
#     tf.summary.histogram(var.name, var)
#
# # Add the loss to summary
# xent_summ = tf.summary.scalar('cross_entropy', xent_loss)
# if opt.siamese: siam_summ = tf.summary.scalar('siamese-loss', siamese_loss)
#
# # Evaluation op: Accuracy of the model
# with tf.name_scope("accuracy"):
#     correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y1, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
#     precision = model.precision
#     recall = model.recall
#     F_alpha = model.F_alpha
#
# # Add the accuracy to the summary
# acc_summ = tf.summary.scalar('accuracy', accuracy)
#
# # Merge all summaries together
# performance = tf.summary.merge([xent_summ, siam_summ]) if opt.siamese else tf.summary.merge([xent_summ, acc_summ])
# merged_summary = tf.summary.merge_all()
#
# # Start Tensorflow session
# with tf.Session() as sess:
#     # Initialize all variables
#     sess.run(tf.global_variables_initializer())
#
#     # Add the model graph to TensorBoard
#     train_writer.add_graph(sess.graph)
#
#     # Load the pretrained weights into the non-trainable layer
#     model.load_model_pretrained(sess)
#
#     print("{} Start training...".format(datetime.now()))
#     print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))
#
#     lowest_loss, lowest_xent, highest_acc = 1e15, 1e15, 0.  # init before checkpointing
#     # Loop over number of epochs
#
#     if opt.siamese:  # if using siamese model to pre-train fc7 layer and run validation
#         for epoch in range(sEpochs):
#             print("{} ------- Siamese Epoch number: {} ------- ".format(datetime.now(), epoch + 1))
#             # initialize before training
#             sess.run(training_init_op)
#             for step in range(aTrainBatches // 2):
#                 img_batch1, label_batch1 = sess.run(next_batch)
#                 img_batch2, label_batch2 = sess.run(next_batch)
#                 # print('label_batches\n', label_batch1, '\n', label_batch2)
#                 sess.run(siamese_train_op, feed_dict={x1: img_batch1,
#                                                       x2: img_batch2,
#                                                       y1: label_batch1,
#                                                       y2: label_batch2,
#                                                       keep_prob: 1 - dropout_rate})
#                 if (epoch * (aTrainBatches // 2) + step) % display_step == 0:
#                     s = sess.run(merged_summary, feed_dict={x1: img_batch1,
#                                                             x2: img_batch2,
#                                                             y1: label_batch1,
#                                                             y2: label_batch2,
#                                                             keep_prob: 1.})
#                     train_writer.add_summary(s, epoch * (aTrainBatches // 2) + step)
#
#             # Validate the model on the entire validation set
#             print("{} Start Validation".format(datetime.now()))
#             sess.run(validation_init_op)
#             test_loss, test_xent, test_count = 0., 0., 0
#             for step in range(aValBatches // 2):
#                 img_batch1, label_batch1 = sess.run(next_batch)
#                 img_batch2, label_batch2 = sess.run(next_batch)
#                 perf, siam, xent = sess.run([performance, siamese_loss, xent_loss], feed_dict={x1: img_batch1,
#                                                                                                x2: img_batch2,
#                                                                                                y1: label_batch1,
#                                                                                                y2: label_batch2,
#                                                                                                keep_prob: 1.})
#                 test_loss += siam * int(label_batch1.shape[0])
#                 test_xent += xent * int(label_batch1.shape[0])
#                 test_count += int(label_batch1.shape[0])
#                 val_writer.add_summary(perf, epoch * (aValBatches // 2) + step)
#             test_loss /= test_count
#             test_xent /= test_count
#             print("{} Validation Loss = {:10f}".format(datetime.now(), float(test_loss)))
#             print("{} Validation Xent = {:10f}".format(datetime.now(), float(test_xent)))
#             sMetrics.update_metrics(('val-loss', 'val-xent'), (test_loss, test_xent))
#             sMetrics.write_metrics()
#
#             # save checkpoint of the model
#             if test_loss < lowest_loss:
#                 lowest_loss = test_loss
#                 print('{} Lowest siamese-loss renewed to {}'.format(datetime.now(), lowest_loss))
#                 if not opt.noCheck:
#                     print('{} Saving checkpoint of the model ...'.format(datetime.now()))
#                     ckpt_path = os.path.join(checkpoint_path, 'model_lowest_siamloss.ckpt')
#                     save_path = saver.save(sess, ckpt_path)
#                     print("{} Model checkpoint saved at {}".format(datetime.now(), ckpt_path))
#             else:  # if test_loss is no better than the current best
#                 print('{} Lowest siamese-loss remained {}'.format(datetime.now(), lowest_loss))
#
#     # Regardless of whether using siamese training, do the following supervised train and val process
#     for epoch in range(aEpochs):
#         print("{} ------- Supervised Training Epoch number: {} ------- ".format(datetime.now(), epoch + 1))
#         # Initialize iterator with the training dataset
#         sess.run(training_init_op)
#         for step in range(aTrainBatches):
#             # get next batch of data
#             img_batch, label_batch = sess.run(next_batch)
#             # And run the training op
#             sess.run(supervised_train_op, feed_dict={x1: img_batch,
#                                                      x2: img_batch,  # x2 is not actually used
#                                                      y1: label_batch,
#                                                      y2: label_batch,
#                                                      keep_prob: 1 - dropout_rate})
#
#             # Generate summary with the current batch of data and write to file
#             if (epoch * aTrainBatches + step) % display_step == 0:
#                 s = sess.run(merged_summary, feed_dict={x1: img_batch,
#                                                         x2: img_batch,  # x2 is not actually used
#                                                         y1: label_batch,
#                                                         y2: label_batch,
#                                                         keep_prob: 1.})
#
#                 train_writer.add_summary(s, epoch * aTrainBatches + step)
#         # Validate the model on the entire validation set
#         print("{} Start Validation".format(datetime.now()))
#         sess.run(validation_init_op)
#         test_acc, test_xent, test_count = 0., 0., 0
#         for step in range(aValBatches):
#             img_batch, label_batch = sess.run(next_batch)
#             perf, acc, xent = sess.run([performance, accuracy, xent_loss], feed_dict={x1: img_batch,
#                                                                                       x2: img_batch,  # x2 is not used
#                                                                                       y1: label_batch,
#                                                                                       y2: label_batch,
#                                                                                       keep_prob: 1.})
#             test_acc += acc * int(label_batch.shape[0])
#             test_xent += xent * int(label_batch.shape[0])
#             test_count += int(label_batch.shape[0])
#             val_writer.add_summary(perf, epoch * aValBatches + step)
#
#         test_acc /= test_count
#         test_xent /= test_count
#         print("{} Validation Accuracy = {}".format(datetime.now(), test_acc))
#         print("{} Validation Cross-Ent = {}".format(datetime.now(), test_xent))
#         aMetrics.update_metrics(('val-acc', 'val-xent'), (test_acc, test_xent))
#         aMetrics.write_metrics()
#
#         # save checkpoint of the model
#         if opt.checkStd == 'xent':  # if the checkpointing standard is lowest cross-entropy, do the following
#             if test_xent < lowest_xent:  # if test_xent is beneath current lowest
#                 lowest_xent = test_xent  # update lowest cross-entropy
#                 print('{} Lowest cross-entropy renewed to {}'.format(datetime.now(), lowest_xent))
#                 if not opt.noCheck:  # skip checkpointing if --noCheck is set
#                     print("{} Saving checkpoint of model...".format(datetime.now()))
#                     ckpt_path = os.path.join(checkpoint_path, 'model_lowest_xent.ckpt')
#                     save_path = saver.save(sess, ckpt_path)
#                     print("{} Model checkpoint saved at {}".format(datetime.now(), ckpt_path))
#             else:  # if test_xent is no better than the current best
#                 print('{} Lowest cross-entropy remained {}'.format(datetime.now(), lowest_xent))
#         elif opt.checkStd == 'acc':  # else, if the checkpointing standard is highest accuracy, do the following
#             if test_acc > highest_acc:  # if test_acc exceeds current highest
#                 highest_acc = test_acc  # update highest accuracy
#                 print('{} Highest accuracy renewed to {}'.format(datetime.now(), highest_acc))
#                 if not opt.noCheck:
#                     print("{} Saving checkpoint of model...".format(datetime.now()))
#                     ckpt_path = os.path.join(checkpoint_path, 'model_highest_acc.ckpt')
#                     save_path = saver.save(sess, ckpt_path)
#                     print("{} Model checkpoint saved at {}".format(datetime.now(), ckpt_path))
#             else:  # if test_acc is no better than the current best
#                 print('{} Highest accuracy remained {}'.format(datetime.now(), highest_acc))
#         else:
#             raise ValueError('Updating mechanism not considered for checkStd ' + opt.checkStd)
pass
# # after training, summarize the embeddings
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(testing_init_op)
#     meta_path = os.path.join(filewriter_path, 'metadata.tsv')
#     with open(meta_path, 'w') as f:
#         f.write('Index\tLabel\n')
#         for step in range(aTestBatches):
#             img_batch, label_batch = sess.run(next_batch)
#             for i, lab in enumerate(list(label_batch)):
#                 f.write('%d\t%d\n' % (step * batch_size1 + i, np.argmax(lab)))
#             fc7_batch, fc8_batch = sess.run([model.fc7, model.fc8], feed_dict={x1: img_batch,
#                                                                                y1: label_batch,
#                                                                                keep_prob: 1.})
#             # concatenate this batch with previous ones
#             if step == 0:
#                 fc7, fc8 = fc7_batch, fc8_batch
#             else:
#                 fc7, fc8 = np.concatenate((fc7, fc7_batch)), np.concatenate((fc8, fc8_batch))
#
#     # checkpoint both fc7 and fc8
#     fc7_var, fc8_var = tf.Variable(fc7, name='fc7_var'), tf.Variable(fc8, name='fc8_var')
#     embedding_saver = tf.train.Saver(var_list=[fc7_var, fc8_var])
#     sess.run([fc7_var.initializer, fc8_var.initializer])
#     config = projector.ProjectorConfig()
#     embedding_fc7, embedding_fc8 = config.embeddings.add(), config.embeddings.add()
#     embedding_fc7.tensor_name, embedding_fc8.tensor_name = fc7_var.name, fc8_var.name
#     embedding_fc7.metadata_path, embedding_fc8.metadata_path = meta_path, meta_path
#
#     # checkpoint only fc8
#     # fc8_var = tf.Variable(fc8, name='fc8_var')
#     # embedding_saver = tf.train.Saver(vars = [fc8_var])
#     # sess.run(fc8_var.initializer)
#     # config = projector.ProjectorConfig()
#     # embedding_fc8 = config.embeddings.add()
#     # embedding_fc8.tensor_name = fc8_var.name
#     # embedding_fc8.metadata_path = meta_path
#
#     projector.visualize_embeddings(tf.summary.FileWriter(filewriter_path), config)
#
#     print('{} Saving embeddings'.format(datetime.now()))
#     embedding_saver.save(sess, os.path.join(filewriter_path, 'embeddings.ckpt'))
#     print('{} Embeddings Saved'.format(datetime.now()))
pass
# print('This is a hanging script, ^C to quit')
# while(True): pass
