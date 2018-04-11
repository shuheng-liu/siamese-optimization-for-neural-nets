"""This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all.

Paper:
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Explanation can be found in my blog post:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of
classes. The structure of this script is strongly inspired by the fast.ai
Deep Learning class by Jeremy Howard and Rachel Thomas, especially their vgg16
finetuning script:
Link:
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py


The pretrained weights can be downloaded here and should be placed in the same
folder as this file:
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""

import numpy as np
import tensorflow as tf


class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(self, x, keep_prob, num_classes, skip_layer, margin=5.0, falpha=2.0,
                 weights_path='/pretrained/bvlc_alexnet.npy'):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.isSiamese = isinstance(x, tuple)
        if self.isSiamese:
            self.X1, self.X2 = x[0], x[1]
        else:
            self.X1, self.X2 = x, x  # self.X2 is never used
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        # with tf.variable_scope('') as scope:

        self.fc7, self.fc8 = self.create(self.X1)
        self.latent1, self.embed1 = self.fc7, self.fc8
        if self.isSiamese:
            tf.get_variable_scope().reuse_variables()
            self.latent2, self.embed2 = self.create(self.X2)
        else:
            self.latent2, self.embed2 = self.fc7, self.fc8

        # define xent_loss
        self.y1 = tf.placeholder(tf.float32, [None, None])
        self.y2 = tf.placeholder(tf.float32, [None, None])  # if not using siamese-training, y2 will not be used
        self.xent_loss = xent_loss(self)
        self.get_precision_recall()
        self.F_alpha = (1 + falpha) / (1 / self.precision + 1 / self.recall)
        if self.isSiamese:
            self.quadratic_siamese_loss = quadratic_siamese_loss(self, margin=margin)
            # self.linear_siamese_loss = linear_siamese_loss(self, margin=margin)
        else:
            self.quadratic_siamese_loss = self.xent_loss
            self.linear_siamese_loss = self.xent_loss

    def create(self, X):
        """Create the network graph. returns tensors of fc7 and fc8"""
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv(X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')
        return fc7, fc8

    def load_initial_weights(self, session):
        """Load weights from file into network.

        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

    def get_precision_recall(self):
        self.TP = tf.reduce_sum(tf.argmax(self.y1) * tf.argmax(self.fc8))
        self.FP = tf.reduce_sum((1 - tf.argmax(self.y1)) * tf.argmax(self.fc8))
        self.FN = tf.reduce_sum(tf.argmax(self.y1) * (1 - tf.argmax(self.fc8)))
        self.FP = tf.reduce_mean((1 - tf.argmax(self.y1)))
        self.precision = self.TP / (self.TP + self.FP)
        self.recall = self.TP / (self.TP + self.FN)


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels / groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)


def quadratic_siamese_loss(net, margin=5.0):
    '''
    This function computes quadratic xent_loss given the pairs (input1, class1) and (input2, class2)
    :param net: the used alexnet instance
    :param margin: the threshold for xent_loss when c1 != c2
    :return: the quadratic xent_loss introduced by the distance between the embeddings of input1 and input2. If they
    have the same class label, the xent_loss is defined by ||embed1 - embed2||; if they have different class labels,
    the xent_loss is defined by ReLU(margin - ||embed1 - embed2||)
    '''
    assert net.isSiamese, 'the model is not a Siamese Network, check again'
    # eucd2 = tf.reduce_sum((net.embed1 - net.embed2) ** 2, name='eucd2')
    eucd2 = tf.reduce_sum((net.latent1 - net.latent2) ** 2, name='eucd2')
    eucd = tf.sqrt(eucd2 + 1e-6, name='eucd')
    y1, y2 = tf.argmax(net.y1), tf.argmax(net.y2)
    y_cmp = tf.cast(y1 - y2, tf.bool)  # xor operation
    print('\n\n\n', y_cmp, '\n\n\n')
    margin = tf.constant(margin, name='margin')
    # if input1 and input2 have the same class label WHICH IS POSITIVE
    # loss1 = tf.multiply(y1, tf.multiply(1. - y_cmp, eucd2), name = 'quad_loss1')
    loss1 = tf.multiply(1. - y_cmp, eucd2)
    loss1 = tf.multiply(y1, loss1, name='quad_loss1')
    # if input1 and input2 have different class labels
    loss2 = tf.multiply(y_cmp, tf.nn.relu(margin - eucd) ** 2, name='quad_loss2')
    loss = tf.reduce_mean(loss1 + loss2, name='reduced_quadloss')
    return loss


def asymmetric_quadratic_siamese_loss(net, margin=5.0):
    assert net.isSiamese, 'the model is not a Siamese Network, check again'
    eucd2 = tf.reduce_mean()


def linear_siamese_loss(net, margin=5.0):
    assert net.isSiamese, 'the model is not a siamese network, check again'
    eucd2 = tf.reduce_sum((net.embed1 - net.embed2) ** 2, name='eucd2')
    eucd = tf.sqrt(eucd2 + 1e-6, name='eucd')
    margin = tf.constant(margin, name='margin')
    # if input1 and input2 share the same class label
    loss1 = tf.pow(tf.multiply(net.y_cmp, eucd), 2, name='loss1')
    # if input1 and input2 have different class labels
    loss2 = tf.pow(tf.multiply(1 - net.y_cmp, (tf.nn.relu(margin - eucd))), 2, name='loss2')
    loss = tf.reduce_mean(loss1 + loss2, name='reduced_linearloss')
    return loss


def xent_loss(net):
    '''returns a cross-entropy xent_loss'''
    with tf.name_scope("cross_ent"):
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=net.fc8, labels=net.y1), name='xent_loss')
