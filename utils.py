import os
import argparse
from datagenerator import ImageDataGenerator


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
    """
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
    """
    print('Auto adapting batch size...')
    numerator = min(train_size, val_size)
    denominator = 0
    while True:
        denominator += batch_count_multiple
        batch_size = numerator // denominator
        if batch_size <= max_size: return batch_size


# make train & val & test list, and do stats
def determine_list(opt, sample_path):
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


# print the info about training, validation and testing sets
def print_info(batch_size, train_batches, val_batches, test_batches, train_length, val_length,
               test_length, model='AlexNet'):
    assert model in ['SiameseAlexNet', 'AlexNet']

    print("********** In model %s **********" % model)
    print('%d samples in training set' % train_length)
    print('%d samples in validation set' % val_length)
    print('Train-Val ratio == %.1f%s : %.1f%s' % (100 * train_length / (train_length + val_length), '%',
                                                  100 * val_length / (train_length + val_length), '%'))
    print('Batch Size =', batch_size)
    print('Of all %d val samples, %d is utilized, percentage = %.1f%s' % (val_length, val_batches * batch_size,
                                                                          val_batches * batch_size / val_length
                                                                          * 100, '%'))
    print('Of all %d train samples, %d is utilized, percentage = %.1f%s' % (train_length, train_batches * batch_size,
                                                                            train_batches * batch_size / train_length
                                                                            * 100, '%'))
    print('Of all %d test samples, %d is utilized, percentage = %.1f%s' % (test_length, test_batches * batch_size,
                                                                           test_batches * batch_size / test_length
                                                                           * 100, '%'))


def get_environment_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train0', required=True, help='paths to negative training dataset, separated by space')
    parser.add_argument('--train1', required=True, help='paths to positive training dataset, separated by space')
    parser.add_argument('--train2', default='', help='paths to other disease training dataset, separated by space')
    parser.add_argument('--trainCeils', default=None, help='Ceils of Training')
    parser.add_argument('--val0', required=True, help='paths to negative validation dataset, separated by space')
    parser.add_argument('--val1', required=True, help='paths to positive validation dataset, separated by space')
    parser.add_argument('--val2', default='', help='paths to other disease validation dataset, separated by space')
    parser.add_argument('--valCeils', default=None, help='Ceils of validation')
    parser.add_argument('--lr1', type=float, default=3e-4, help='learning rate for supervised learning, default=1e-3')
    parser.add_argument('--lr2', type=float, default=5e-7, help='learning rate for siamese learning, default=5e-7')
    parser.add_argument('--nepochs1', type=int, default=100, help='number of supervised epochs, default = 100')
    parser.add_argument('--nepochs2', type=int, default=100, help='number of siamese epochs, default = 100')
    parser.add_argument('--batchSize1', type=int, default=0, help='default = automatic-adapting')
    parser.add_argument('--batchSize2', type=int, default=0, help='default = automatic-adapting')
    parser.add_argument('--dropout', type=int, default=0.5, help='dropout rate for alexnet, default = 0.5')
    parser.add_argument('--nclasses', type=int, default=2, help='number of classes, default = 2')
    parser.add_argument('--trainLayers1', type=str, default='fc7 fc8', help='default = fc7 fc8')
    parser.add_argument('--trainLayers2', type=str, default='fc6', help='default = fc6')
    parser.add_argument('--displayStep', type=int, default=200, help='How often to write tf.summary')
    parser.add_argument('--outf', type=str, default='/output', help='path for checkpoints & tf.summary & samplelist')
    parser.add_argument('--pretrained', type=str, default='/', help='path for pre-trained weights *.npy')
    parser.add_argument('--noCheck', action='store_true', help='don\'t save model checkpoints')
    parser.add_argument('--siamese', type=str, default='dropout6', help='siamese projection layers, default=dropout6')
    parser.add_argument('--checkStd', type=str, default='xent', help='Standard for checkpointing, acc or xent')
    parser.add_argument('--margin00', type=float, default=8.0, help='distance margin for neg-neg pair, default=10.0')
    parser.add_argument('--margin11', type=float, default=6.0, help='distance margin for pos-pos pair, default=8.0')
    parser.add_argument('--margin01', type=float, default=7.07, help='distance margin for neg-pos pair, default=7.0')
    parser.add_argument('--punish00', type=float, default=1.0, help='punishment for neg-neg pair, default=1.0')
    parser.add_argument('--punish11', type=float, default=1.0, help='punishment for pos-pos pair, default=1.0')
    parser.add_argument('--punish01', type=float, default=5.0, help='punishment for neg-pos pair, default=5.0')
    return parser.parse_args()


def get_init_op(iterator, some_data: ImageDataGenerator):
    return iterator.make_initializer(some_data.data)


def get_precision_recall_fscore(TP=0, TN=0, FP=0, FN=0, alpha=1.0):
    # convert to int instead of np.int32
    TP, TN, FP, FN = int(TP), int(TN), int(FP), int(FN)

    # get precision
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        print('ZeroDivisionError in calculating precision')
        precision = 0

    # get recall
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        print('ZeroDivisionError in calculating recall')
        recall = 0

    # get F score
    alpha_squared = alpha ** 2  # calculate alpha^2 in advance
    try:
        fscore = (1 + alpha_squared) * TP / ((1 + alpha_squared) * TP + alpha_squared * FN + FP)
    except ZeroDivisionError:
        print('ZeroDivisionError in calculating F score')
        fscore = 0

    return precision, recall, fscore
