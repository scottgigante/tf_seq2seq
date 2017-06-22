
# coding: utf-8

# In[155]:

# Code gleaned and modified from https://medium.com/towards-data-science/text-summarization-with-amazon-reviews-41801c2210b

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import os
import h5py
import glob

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.layers import safe_embedding_lookup_sparse as embedding_lookup_unique
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
"""

import numpy as np
import tensorflow as tf

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from NanoporeData import NanoporeData
from FakeNanoporeData import FakeNanoporeData
from NanoporeModel import NanoporeModel
from NanoporeSeq2SeqTrainer import NanoporeSeq2SeqTrainer


#get_ipython().magic('matplotlib inline')

# Constants
seed=42

# Input data
simulate=False
# Simulated
num_classes = 4 # Nanopore data has exactly four classes, A, C, G AND T.
num_features = 1024 # Fix input sequence length.
# Real
data_dir = "nanopore_data"
max_files = 1000

# Input size
batch_size = 16 # Number of inputs per batch
num_examples = 500 if not simulate else 8192 * 16 # Number of inputs per epoch

# Model building
num_encode_layers = 1
conv_widths = [3, 7, 15]
conv_size = 32
output_embedding_size = 16
num_decode_layers = 2
rnn_size = 64

# Model training
epochs = 200
max_tests_no_best = 50
learning_rate = 1e-4
min_learning_rate = 1e-6
learning_rate_decay = 0.95
sample_prob = 0.0
sample_prob_decay = 0.95
min_sample_prob = 0.95
keep_probability = 0.7
length_cost_prop = 0.5

# Training display
update_per_epoch = 20
display_per_epoch = update_per_epoch*4

checkpoint = "best_model.ckpt"


# In[209]:

np.random.seed(seed)
if simulate:
    data = FakeNanoporeData(batch_size, num_examples, num_features, num_classes)
else:
    data = NanoporeData(data_dir, batch_size, max_files = num_examples)

if False:
    # plot input distribution
    fig = plt.figure()
    if simulate:
        ax1 = fig.add_subplot(211)
        x = np.linspace(300, 420, 100)
        for mu, sigma in data.model.values():
            ax1.plot(x,mlab.normpdf(x, mu, sigma))
        ax2 = fig.add_subplot(212)
    else:
        ax2 = fig.add_subplot(111)
    keys, values = zip(*data.get_input_vocab_freq().items())
    ax2.bar(keys, values)
    if simulate:
        ax2.set_xlim(ax1.get_xlim())
    fig.savefig("vocab_dist.png", bbox_inches='tight')
    #plt.show()


# In[210]:
logging.info("Building network...")
model = NanoporeModel(data.input_embedding_matrix,
                      num_classes,
                      batch_size,
                      output_embedding_size,
                      conv_size,
                      conv_widths,
                      num_encode_layers,
                      rnn_size,
                      num_decode_layers,
                      seed=seed)


# In[211]:

trainer = NanoporeSeq2SeqTrainer(model,
                                 learning_rate,
                                 min_learning_rate,
                                 learning_rate_decay,
                                 sample_prob,
                                 sample_prob_decay,
                                 min_sample_prob,
                                 keep_probability,
                                 length_cost_prop,
                                 epochs,
                                 max_tests_no_best,
                                 verbose=True,
                                 save_best=True,
                                 display_per_epoch=display_per_epoch,
                                 update_per_epoch=update_per_epoch)


# In[122]:

logging.info("Beginning training.")
if False: # parameter search
    model_params, learn_params = trainer.parameter_search(data, model_params={
            'num_encode_layers' : [1, 2, 3],
            'conv_size' : [8, 16, 32, 64, 128],
            'conv_widths' : [[3, 7], [7, 15], [15, 31], [3, 7, 15], [7, 15, 31], [3, 7, 15, 31]],
            'output_embedding_size' : [4, 8, 16, 32, 64],
            }, learn_params={
            'learning_rate' : [0.001, 0.005, 0.01, 0.05, 0.1],
            'keep_probability' : [0.6, 0.7, 0.8],
            'learning_rate_decay' : [0.98, 0.95, 0.9, 0.8],
            })
else:
    loss = trainer.run_model_training(data)


outputs_batch, inputs_batch, outputs_lengths, inputs_lengths = next(data.get_test_batches())
train_outputs_batch, train_inputs_batch, train_outputs_lengths, train_inputs_lengths = next(data.get_train_batches())

loaded_graph = model.train_graph
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    #loader = tf.train.import_meta_graph("seq_" + checkpoint + '.meta')
    model.saver.restore(sess, "seq_" + checkpoint)
    logging.info("Testing predictions...")
    train_predictions = sess.run(model.decoder.predictions, {model.input_data: train_inputs_batch,  
                                      model.text_length: train_inputs_lengths,
                                      model.keep_prob: 1.0})
    
    test_predictions = sess.run(model.decoder.predictions, {model.input_data: inputs_batch, 
                                      model.text_length: inputs_lengths,
                                      model.keep_prob: 1.0})

print("Training data:")
print("Real:", train_outputs_batch)
print("Pred:", train_predictions)
print("Test data:")
print("Real:", outputs_batch)
print("Pred:", test_predictions)
