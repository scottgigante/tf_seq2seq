import tensorflow as tf

class Optimizer:
    __max_grad = 5.
    def __init__(self, learning_rate, name="optimizer"):
        with tf.name_scope("optimization"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate, name=name)
    
    def apply_gradients(self, cost):
        with tf.name_scope("optimization"):
            gradients = self.optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -self.__max_grad, 
                    self.__max_grad), var) for grad, var in gradients if grad is not None]
        return self.optimizer.apply_gradients(capped_gradients)
