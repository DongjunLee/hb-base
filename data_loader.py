# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import os

import numpy as np
from hbconfig import Config
import tensorflow as tf




def process_data():
    print('Preprocessing data to be model-ready ...')

    # create path to store all the train & test encoder & decoder
    os.mkdir(Config.data.base_path + Config.data.processed_path)

    preprocessing()


def preprocessing():
    # TODO: implements logic
    pass


def read_data():
    # TODO: implements logic
    pass


def make_train_and_test_set(shuffle=True):
    print("make Training data and Test data Start....")

    # load train and test dataset
    train_X, train_y = read_data()
    test_X, test_y = read_data()

    assert len(train_X) == len(train_y)
    assert len(test_X) == len(test_y)

    print(f"train data count : {len(train_y)}")
    print(f"test data count : {len(test_y)}")

    if shuffle:
        print("shuffle dataset ...")
        train_p = np.random.permutation(len(train_y))
        test_p = np.random.permutation(len(test_y))

        return ((train_X[train_p], train_y[train_p]),
                (test_X[test_p], test_y[test_p]))
    else:
        return ((train_X, train_y),
                (test_X, test_y))


def make_batch(X, y, buffer_size=10000, batch_size=64, scope="train"):

    class IteratorInitializerHook(tf.train.SessionRunHook):
        """Hook to initialise data iterator after Session is created."""

        def __init__(self):
            super(IteratorInitializerHook, self).__init__()
            self.iterator_initializer_func = None

        def after_create_session(self, session, coord):
            """Initialise the iterator after the session has been created."""
            self.iterator_initializer_func(session)


    iterator_initializer_hook = IteratorInitializerHook()

    def inputs():
        with tf.name_scope(scope):

            # Define placeholders
            input_placeholder = tf.placeholder(
                tf.int32, [None] * len(X.shape), name="input_placeholder")
            target_placeholder = tf.placeholder(
                tf.int32, [None] * len(y.shape), name="target_placeholder")

            # Build dataset iterator
            dataset = tf.data.Dataset.from_tensor_slices(
                (input_placeholder, target_placeholder))

            if scope == "train":
                dataset = dataset.repeat(None)  # Infinite iterations
            else:
                dataset = dataset.repeat(1)  # one Epoch

            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            next_input, next_target = iterator.get_next()

            tf.identity(next_input[0], 'input_0')
            tf.identity(next_target[0], 'target_0')

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: X,
                               target_placeholder: y})

            # Return batched (features, labels)
            return next_input, next_target

    # Return function and hook
    return inputs, iterator_initializer_hook




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    process_data()
