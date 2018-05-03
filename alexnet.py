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


class Model(object):
    def __init__(self):
        pass

    def set_model_vars(self, variable_dict, session):
        pass

    def get_model_vars(self, session, init=False):
        return {}

    def load_model_vars(self, path: str, session):
        pass

    def save_model_vars(self, path: str, session, init=False):
        pass

    def load_model_pretrained(self, session):
        pass

    def _create_loss(self, *args):
        pass


# noinspection PyCompatibility
class AlexNet(Model):
    """Implementation of the AlexNet."""
    TRAIN_LAYERS = ...  # type: set
    y = ...  # type: tf.placeholder

    # TODO ATTENTION: loading pretrained weights is called outside the constructor
    def __init__(self, x, keep_prob, num_classes, train_layers, falpha=2.0,
                 weights_path='/pretrained/bvlc_alexnet.npy'):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            train_layers: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        super(AlexNet, self).__init__()
        self.X = x
        # self.X = tf.placeholder(tf.float32, [None, 227, 227, 3])
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.TRAIN_LAYERS = train_layers
        self.WEIGHTS_PATH = weights_path
        self.ALPHA = falpha

        # Call the create function to build the computational graph of AlexNet
        # with tf.variable_scope('') as scope:

        self._create_discriminator()

        # define metrics
        # TODO consider switching the second dimension to self.NUM_CLASSES
        self.y = tf.placeholder(tf.float32, [None, self.NUM_CLASSES], name='y')
        self.correct_pred = tf.equal(tf.argmax(self.fc8, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')

        self._create_loss()
        self._create_stats(falpha)

    def _create_discriminator(self):
        """Create the network graph. returns tensors of fc7 and fc8"""
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
        self.conv1, self.norm1, self.pool1 = conv1, norm1, pool1

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
        self.conv2, self.norm2, self.pool1 = conv2, norm2, pool2

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')
        self.conv3 = conv3

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
        self.conv4 = conv4

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        self.conv5, self.pool5 = conv5, pool5

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB, name='dropout6')
        self.flattened, self.fc6, self.dropout6 = flattened, fc6, dropout6

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB, name='dropout7')
        self.fc7, self.dropout7 = fc7, dropout7

        # 8th Layer: FC and return unscaled activations
        fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')
        self.fc8 = fc8

    def _create_loss(self):
        with tf.name_scope("cross_ent"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc8, labels=self.y),
                                       name="loss")

    def load_model_pretrained(self, session):
        """Load weights from file into network.

        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        variable_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()  # type: dict
        # Loop over all layer names stored in the weights dict
        for op_name in variable_dict:  # type: str
            # Check if layer should be trained from scratch
            if op_name not in self.TRAIN_LAYERS:
                with tf.variable_scope(op_name, reuse=True):
                    # Assign weights/biases to their corresponding tf variable
                    for data in variable_dict[op_name]:
                        var_name = "biases" if len(data.shape) == 1 else "weights"
                        var = tf.get_variable(var_name, trainable=False)
                        try:
                            session.run(var.assign(data))
                        except:
                            print("Failed to assign value to", var.name)

    # TODO debug _create_loss, precision and F_score is somehow always NaN, maybe due to wrong choice of axis in argmax
    def _create_stats(self, alpha):
        """only works for binary classification"""
        # only works for binary classification
        prediction = tf.argmax(self.fc8, axis=1, name='alexnet-prediction')
        ground_truth = tf.argmax(self.y, axis=1, name='alexnet-ground-truth')
        self.prediction, self.ground_truth = prediction, ground_truth
        self.TP = tf.reduce_sum(prediction * ground_truth)  # True Positive
        self.TN = tf.reduce_sum((1 - prediction) * (1 - ground_truth))  # True Negative
        self.FP = tf.reduce_sum(prediction * (1 - ground_truth))  # False Positive
        self.FN = tf.reduce_sum((1 - prediction) * ground_truth)  # False Negative
        self.precision = self.TP / (self.TP + self.FP)
        self.recall = self.TP / (self.TP + self.FN)
        self.F_alpha = (1 + alpha) / (1 / self.precision + alpha / self.recall)
        if self.NUM_CLASSES != 2:
            print("Warning: precision, recall and F_alpha score does not apply to Multi-Label Classification")

    def get_model_vars(self, session, init=False):
        if init: session.run(tf.global_variables_initializer())
        layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
        variable_dict = {layer: [] for layer in layers}
        for layer in variable_dict:
            with tf.variable_scope(layer, reuse=True):
                for var_name in ["weights", "biases"]:
                    var = tf.get_variable(var_name)
                    variable_dict[layer].append(session.run(var))
        return variable_dict

    def set_model_vars(self, variable_dict, session):
        for op_name in variable_dict:
            with tf.variable_scope(op_name, reuse=True):
                for data in variable_dict[op_name]:
                    var_name = 'biases' if len(data.shape) == 1 else "weights"
                    # in case set_model_vars() is called before load_model_pretrained(), set trainable
                    var = tf.get_variable(var_name, trainable=op_name in self.TRAIN_LAYERS)
                    session.run(var.assign(data))

    def save_model_vars(self, path: str, session, init=False):
        np.save(path, self.get_model_vars(session, init=init))

    def load_model_vars(self, path: str, session):
        variable_dict = np.load(path, encoding="bytes").item()  # type: dict
        self.set_model_vars(variable_dict, session)


class SiameseAlexNet(Model):
    # TODO ATTENTION: loading pretrained weights is called outside the constructor
    def __init__(self, x1, x2, keep_prob, num_classes, train_layers, name_scope="Siamese", proj="flattened",
                 falpha=2.0, margin=5.0, weights_path='/pretrained/bvlc_alexnet.npy'):
        super(SiameseAlexNet, self).__init__()
        self.name_scope = name_scope
        self.margin = margin
        self.proj = proj
        with tf.variable_scope(self.name_scope) as scope:
            self.net1 = AlexNet(x1, keep_prob, num_classes, train_layers, falpha=falpha, weights_path=weights_path)
            scope.reuse_variables()
            self.net2 = AlexNet(x2, keep_prob, num_classes, train_layers, falpha=falpha, weights_path=weights_path)
            # define a loss for Siamese Network
            self._create_loss(proj)

    def _create_loss(self, proj):
        proj1, proj2 = self._get_projections(proj)
        eucd2 = tf.reduce_sum((proj1 - proj2) ** 2, name="euclidean_dist_squared")
        eucd = tf.sqrt(eucd2, name="euclidean_dist")
        # y1, y2 and y_cmp should be wrapped instead of being a class member
        y1 = tf.argmax(self.net1.y, axis=1, name='siam-y1')
        y2 = tf.argmax(self.net2.y, axis=1, name='siam-y2')
        self.y1_label, self.y2_label = y1, y2
        y_diff = tf.cast(y1 - y2, tf.bool, name="comparison_label_in_tf.bool")
        y_diff = tf.cast(y_diff, tf.float32, name="comparison_label_in_tf.float32")
        self.diff_count = tf.reduce_sum(y_diff, name='diff_count')
        self.same_count = tf.reduce_sum(tf.cast(y1 == y2, tf.int32), name='same_count')
        self.same_count = tf.reduce_sum(tf.cast(tf.equal(y1, y2), tf.float32))
        margin = tf.constant(self.margin, name="margin")
        # if label1 and label2 are the same, y_diff = 0, punish the part where eucd exceeds margin
        loss_same = tf.reduce_mean(tf.multiply(1 - y_diff, tf.nn.relu(eucd - margin)) ** 2, name='loss_same')
        # if label1 and label2 are different, y_diff = 1, punish the part where eucd falls short of margin
        loss_diff = tf.reduce_mean(tf.multiply(y_diff, tf.nn.relu(margin - eucd)) ** 2, name='loss_diff')
        self.loss = tf.add(loss_same, loss_diff, name="siamese-loss")
        print(self.loss)

    def _get_projections(self, proj):
        print('projection =', proj, "type=", type(proj))
        projections = (self.net1.dropout6, self.net2.dropout6)
        try:
            if proj == "fc6":
                projections = (self.net1.fc6, self.net2.fc6)
            elif proj == "fc7":
                projections = (self.net1.fc7, self.net2.fc7)
            elif proj == "fc8":
                projections = (self.net1.fc8, self.net2.fc8)
            elif proj == "dropout6":
                projections = (self.net1.dropout6, self.net2.dropout6)
            elif proj == "dropout7":
                projections = (self.net1.dropout7, self.net2.dropout7)
            elif proj == "flattened":
                projections = (self.net1.flattened, self.net2.flattened)
            else:
                raise ValueError("Illegal Projection: " + proj)
        except ValueError as e:
            print(e)
        finally:
            print("projections of %s are " % self.name_scope, projections[0].name, projections[1].name)
            print("dimensions of projection is", projections[0].shape, projections[1].shape)
            return projections

    def load_model_pretrained(self, session):
        with tf.variable_scope(self.name_scope, reuse=True):
            self.net1.load_model_pretrained(session)

    def load_model_vars(self, path: str, session):
        with tf.variable_scope(self.name_scope, reuse=True):
            self.net1.load_model_vars(path, session)

    def save_model_vars(self, path: str, session, init=False):
        with tf.variable_scope(self.name_scope):
            self.net1.save_model_vars(path, session, init=init)

    def get_model_vars(self, session, init=False):
        with tf.variable_scope(self.name_scope):
            return self.net1.get_model_vars(session, init=init)

    def set_model_vars(self, variable_dict, session):
        with tf.variable_scope(self.name_scope):
            return self.net1.set_model_vars(variable_dict, session)

    # return a new instance of AlexNet with trainable variables
    def get_net_copy(self, session, x=None, keep_prob=None, num_classes=None, train_layers=None, falpha=None,
                     weights_path=None) -> AlexNet:
        if x is None:
            x = self.net1.X
            print("Warning: x should be specified as a new placeholder")
        if keep_prob is None:
            keep_prob = self.net1.KEEP_PROB
        if num_classes is None:
            num_classes = self.net1.NUM_CLASSES
        if train_layers is None:
            train_layers = self.net1.TRAIN_LAYERS
            print("Warning: train_layers should be specified as a new list of layer names")
        if falpha is None:
            falpha = self.net1.ALPHA
        if weights_path is None:
            weights_path = self.net1.WEIGHTS_PATH
        new_net = AlexNet(x, keep_prob, num_classes, train_layers, falpha=falpha, weights_path=weights_path)
        new_net.set_model_vars(self.get_model_vars(session), session)
        return new_net


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
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


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob, name='dropout'):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob, name=name)


if __name__ == "__main__":
    # TODO how does the two nets in Siamese Net share the keep_prob placeholder?
    keep_prob = tf.placeholder(tf.float32, [], name = 'keep_prob')
    x = tf.placeholder(tf.float32, [None, 227, 227, 3], name='x')
    # x1 = tf.placeholder(tf.float32, [None, 227, 227, 3], name='x1')
    # x2 = tf.placeholder(tf.float32, [None, 227, 227, 3], name='x2')
    # image_batch = np.random.rand(5, 227, 227, 3)
    # label_batch = np.random.rand(5, 1000)
    net = AlexNet(x, keep_prob, 3, ['fc6', 'fc7'])
    # net = SiameseAlexNet(x1, x2, 0.5, 3, ['fc6', 'fc7', 'fc8'], name_scope="SiameseA", proj="flattened")
    # netB = SiameseAlexNet(x1, x2, 0.5, 3, ['fc6', 'fc7', 'fc8'], name_scope="SiameseB")
    # check_path = "/Users/liushuheng/Desktop/vars.npy"
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        net.load_model_pretrained(sess)
    # y1 = sess.run(netA.net1.y, feed_dict={netA.net1.X: image_batch, netA.net1.y: label_batch})
    # y2 = sess.run(netB.net1.y, feed_dict={netB.net1.X: image_batch, netB.net1.y: label_batch})
    # netA.save_model_vars(check_path, sess)
    # netB.load_model_vars(check_path, sess)
    # y3 = sess.run(netB.net1.y, feed_dict={netB.net1.X: image_batch, netB.net1.y: label_batch})
    # assert (y1 == y2).all(), "assertion1 failed"
    # print("assertion1 passed")
    # assert (y1 == y3).all(), "assertion2 failed"
    # print("assertion2 passed")
    # d = net.get_model_vars(sess)
    # init_weights = np.load("/pretrained/bvlc_alexnet.npy", encoding="bytes").item()

    # for var in tf.global_variables():
    # # for var in tf.get_default_graph().get_operations():
    #     print(var.name, end=" ")
