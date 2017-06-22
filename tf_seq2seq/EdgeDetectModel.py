import tensorflow as tf

from Optimizer import Optimizer

class EdgeDetectModel:
    
    def __init__(self, inputs, targets, learning_rate, batch_size):
        self.batch_size = batch_size
        with tf.variable_scope("edge_detection"):
            self.output = self.edge_detection(inputs)
            self.cost = self.calculate_loss(self.output, targets)
            self.optimizer = Optimizer(learning_rate)
            self.train_op = self.optimizer.apply_gradients(self.cost)
        self.summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='edge_detection'))
        
    def edge_detection(self, enc_output):
        with tf.name_scope("output"):
            dense_cell = tf.layers.dense(inputs=enc_output, units=1, activation=tf.nn.tanh)
            edge_detect = tf.reduce_mean(dense_cell, axis=2)
        return edge_detect
    
    def mask_targets(self, targets, batch_size, as_float=False):
        # replace the first value with 0 no matter what
        if as_float:
            pad = 0.0
        else:
            pad = 0
        ending = tf.slice(targets, [0, 1], [-1, -1])
        masked_targets = tf.concat([tf.fill([batch_size, 1], pad), ending], 1)
        return masked_targets
    
    def calculate_loss(self, predictions, targets):
        with tf.name_scope("loss"):
            masked_predictions = self.mask_targets(predictions, self.batch_size, as_float=True)
            masked_targets = self.mask_targets(targets, self.batch_size)
            cost = tf.losses.mean_squared_error(masked_predictions, masked_targets)
            tf.summary.scalar("loss", cost)
        return cost
