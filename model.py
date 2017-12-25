from __future__ import print_function


from hbconfig import Config
import tensorflow as tf

import concrete_model



class Model:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.dtype = tf.float32

        self.mode = mode
        self.params = params

        self._init_placeholder(features, labels)
        self.build_graph()

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=self.loss,
                train_op=self.train_op)

        elif self.mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=self.loss,
                eval_metric_ops=self._build_metric())

        elif self.mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=self.predictions)

    def _init_placeholder(self, features, labels):
        self.inputs = features
        if type(features) == dict:
            self.inputs = features["input_data"]
        self.targets = labels

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            # TODO: define train placeholder
            pass
        else:
            # TODO: define eval/predict placeholder
            pass


    def build_graph(self):
        graph = concrete_model.Graph(self.mode)
        output = graph.build(inputs=self.inputs)

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self._build_loss(output)
            self._build_optimizer()
        else:
            # TODO: define eval/predict graph
            pass

    def _build_loss(self, logits):
        with tf.variable_scope('loss'):
            # TODO: self.loss
            pass

    def _build_optimizer(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer=Config.train.get('optimizer', 'Adam'),
            learning_rate=Config.train.learning_rate,
            summaries=['loss', 'gradients', 'learning_rate'],
            name="train_op")

    def _build_metric(self):
        # TODO: implements tf.metrics
        #   example) return {"accuracy": tf.metrics.accuracy(labels, predicitions)}
        return {}

