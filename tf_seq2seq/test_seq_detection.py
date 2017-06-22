
import tensorflow as tf
import numpy as np

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from NanoporeData import NanoporeData
from FakeNanoporeData import FakeNanoporeData
from NanoporeModel import NanoporeModel

# Constants
seed=42

# Input data
simulate=True
# Simulated
num_classes = 4 # Nanopore data has exactly four classes, A, C, G AND T.
num_features = 1024 # Fix input sequence length.
# Real
data_dir = "nanopore_data"

# Input size
batch_size = 64 # Number of inputs per batch
num_examples = 50 if not simulate else 8192 * 16 # Number of inputs per epoch

# Model building
num_encode_layers = 1
conv_widths = [3, 7, 15]
conv_size = 32
output_embedding_size = 16
num_decode_layers = 2
rnn_size = 64

checkpoint = "best_model.ckpt"


# In[209]:

np.random.seed(seed)
if simulate:
    data = FakeNanoporeData(batch_size, num_examples, num_features, num_classes)
else:
    data = NanoporeData(data_dir, batch_size, max_files = num_examples)
    
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

np.set_printoptions(threshold=np.inf)
print("Training data:")
for i in range(len(train_outputs_batch)):
    print("Real:", train_outputs_batch[i])
    print("Pred:", train_predictions[i])
print("Test data:")
for i in range(len(outputs_batch)):
    print("Real:", outputs_batch[i])
    print("Pred:", test_predictions[i])
