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

import os, argparse

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
from tensorflow.contrib.tensorboard.plugins import projector
import math
"""
Configuration Part.
"""

def make_list(folders, flags = None, ceils = None, mode = 'train', store_path = '/output'):
    suffices = ('jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG')
    if ceils is None: ceils = [-1] * len(folders) # ceil constraint not imposed
    if flags is None: flags = list(range(len(folders))) # flags = [0, 1, ..., n-1]
    assert len(folders) == len(flags) == len(ceils)
    assert mode in ['train', 'val', 'test']
    for folder in folders: assert os.path.isdir(folder), "%s is not a directory" % folder

    print('Making %s list' % mode)
    if not os.path.isdir(store_path): os.mkdir(store_path)
    out_list = os.path.join(store_path, mode + '.txt')
    list_length = 0
    with open(out_list, 'w') as fo:
        for folder, flag, ceil in zip(folders, flags, ceils):
            count = 0
            for pic_name in os.listdir(folder):
                if pic_name.split('.')[-1] not in suffices:
                    print('Ignoring non-image file %s in folder %s' % (pic_name, folder))
                    print('Legal prefices are {}'.format(suffices))
                    continue
                count += 1
                list_length += 1
                fo.write("{} {}\n".format(os.path.join(folder, pic_name), flag))
                # if ceil is imposed (ceil > 0) and count exceeds ceil, break and write next flag
                if 0 < ceil <= count: break
    print('%s list made' % mode)
    return out_list, list_length

parser = argparse.ArgumentParser()
parser.add_argument('--train0', required=True, help='path to negative training dataset')
parser.add_argument('--train1', required=True, help='path to positive training dataset')
parser.add_argument('--val0', required=True, help='path to negative validation dataset')
parser.add_argument('--val1', required=True, help='path to positive validation dataset')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default = 0.001')
parser.add_argument('--nepochs', type=int, default=20, help='number of epochs, default = 20')
parser.add_argument('--batchSize', type=int, default=128, help='default = 128')
parser.add_argument('--dropout', type=int, default=0.5, help='dropout rate for alexnet, default = 0.5')
parser.add_argument('--nclasses', type=int, default=2, help='number of classes, default = 2')
parser.add_argument('--trainLayers', type=str, default='fc8 fc7 fc6', help='default = fc6 ~ fc8')
parser.add_argument('--displayStep', type=int, default=20, help='How often to write tf.summary')
parser.add_argument('--outf', type=str, default='/output', help='path for checkpoints & tf.summary')
parser.add_argument('--pretrained', type=str, default = '/', help='path for pre-trained weights *.npy')
parser.add_argument('--noCheck', action = 'store_true', help='don\'t save model checkpoints')
parser.add_argument('--checkStd', type=str, default='xent', help='Standard for checkpointing, acc or xent')
opt = parser.parse_args()
print(opt)

train_file, train_length = make_list((opt.train0, opt.train1), mode='train', store_path=opt.outf)
val_file, val_length = make_list((opt.val0, opt.val1), mode='val', store_path=opt.outf)

# Learning params
learning_rate = opt.lr
num_epochs = opt.nepochs
batch_size = opt.batchSize

# Network params
dropout_rate = opt.dropout
num_classes = opt.nclasses
train_layers = opt.trainLayers.split()

# How often we want to write the tf.summary data to disk
display_step = opt.displayStep
assert opt.checkStd in ['acc', 'xent'], 'Illegal check standard, %s' % opt.checkStd

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = os.path.join(opt.outf, 'tensorboard')
checkpoint_path = os.path.join(opt.outf, 'checkpoints')

print('%d samples in training set' % train_length)
print('%d samples in validation set' % val_length)
print('Train - val ratio == %.1f%s : %.1f%s' % (100 * train_length / (train_length + val_length), '%',
                                              100 * val_length / (train_length + val_length), '%'))
print('Of all %d val samples, %d is utilized, percentage = %.1f%s' % (val_length,
                                            val_length // batch_size * batch_size,
                                            val_length // batch_size * batch_size / val_length * 100, '%') )

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()
print('data loaded and preprocessed on the cpu')
# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
xent_summ = tf.summary.scalar('cross_entropy', loss)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
acc_summ = tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
performance = tf.summary.merge([xent_summ, acc_summ])
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
train_writer = tf.summary.FileWriter(os.path.join(filewriter_path, 'train'))
val_writer = tf.summary.FileWriter(os.path.join(filewriter_path, 'val'))

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = math.floor(tr_data.data_size / batch_size)
val_batches_per_epoch = math.floor(val_data.data_size / batch_size)

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    train_writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))


    lowest_xent, highest_acc = 99999., 0. # init before checkpointing
    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if (epoch*train_batches_per_epoch + step) % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                train_writer.add_summary(s, epoch*train_batches_per_epoch + step)

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_xent = 0.
        test_count = 0
        for step in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            perf, acc, xent = sess.run([performance, accuracy, loss], feed_dict={x: img_batch,
                                                                     y: label_batch,
                                                                     keep_prob: 1.})
            test_acc += acc * int(label_batch.shape[0])
            test_xent += xent * int(label_batch.shape[0])
            test_count += int(label_batch.shape[0])
            val_writer.add_summary(perf, epoch * val_batches_per_epoch + step)
        test_acc /= test_count
        test_xent /= test_count
        print("{} Validation Accuracy = {}".format(datetime.now(), test_acc))
        print("{} Validation Cross-Ent = {}".format(datetime.now(), test_xent))

        if opt.noCheck: continue # skip checkpointing
        # save checkpoint of the model
        if opt.checkStd == 'xent': # if the checkpointing standard is lowest cross-entropy, do the following
            if test_xent < lowest_xent: # if test_xent is beneath current lowest
                lowest_xent = test_xent # update lowest cross-entropy
                print('{} Lowest cross-entropy renewed to {}'.format(datetime.now(), lowest_xent))
                print("{} Saving checkpoint of model...".format(datetime.now()))
                checkpoint_name = os.path.join(checkpoint_path, 'model_lowest_xent.ckpt')
                save_path = saver.save(sess, checkpoint_name)
                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
            else: # if test_xent is no better than the current best
                print('{} Lowest cross-entropy remained {}'.format(datetime.now(), lowest_xent))
        else: # else, if the checkpointing standard is highest accuracy, do the following
            if test_acc > highest_acc: # if test_acc exceeds current highest
                highest_acc = test_acc # update highest accuracy
                print('{} Highest accuracy renewed to {}'.format(datetime.now(), highest_acc))
                print("{} Saving checkpoint of model...".format(datetime.now()))
                checkpoint_name = os.path.join(checkpoint_path, 'model_highest_acc.ckpt')
                save_path = saver.save(sess, checkpoint_name)
                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
            else: # if test_acc is no better than the current best
                print('{} Highest accuracy remained {}'.format(datetime.now(), highest_acc))


train_writer.close()
val_writer.close()
# print('The program has already finished. There is a dead loop at the end, ctrl+c to quit.')
# while(True): pass