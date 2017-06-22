import tensorflow as tf

from Optimizer import Optimizer

class MeanDetectModel:
    
    def __init__(self, inputs, targets, learning_rate, min_mean, max_mean):
        self.min_mean = min_mean
        self.max_mean = max_mean
        with tf.variable_scope("mean_detection"):
            logits = self.mean_detection(inputs)
            self.output = self.inverse_scale_mean_targets(logits)
            self.cost = self.calculate_loss(logits, targets)
            self.optimizer = Optimizer(learning_rate)
            self.train_op = self.optimizer.apply_gradients(self.cost)
        self.summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='mean_detection'))
        
    def mean_detection(self, enc_output):
        with tf.name_scope("output"):
            dense_cell = tf.layers.dense(inputs=enc_output, units=1) # linear activation
            means = tf.reduce_mean(dense_cell, axis=2)
        return means
        
    def scale_mean_targets(self, targets):
        means_range = self.max_mean - self.min_mean
        return (targets - self.min_mean) * 2 / means_range - 1
    
    def inverse_scale_mean_targets(self, targets):
        means_range = tf.cast(self.max_mean - self.min_mean, tf.float32)
        return (targets + 1.) * means_range / 2. + tf.cast(self.min_mean, tf.float32)
        
    def calculate_loss(self, predictions, targets):
        with tf.name_scope("loss"):
            scaled_targets = self.scale_mean_targets(targets)
            cost = tf.losses.mean_squared_error(predictions, scaled_targets)
        tf.summary.scalar("loss", cost)
        return cost
