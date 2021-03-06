
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

logging.info("Testing predictions...")

outputs_batch, inputs_batch, outputs_lengths, inputs_lengths = next(data.get_test_batches(label_means=True))
train_outputs_batch, train_inputs_batch, train_outputs_lengths, train_inputs_lengths = next(data.get_train_batches(label_means=True))

loaded_graph = model.train_graph
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    #loader = tf.train.import_meta_graph(checkpoint + '.meta')
    
    
    model.saver.restore(sess, "mean_" + checkpoint)
    
    train_predictions = sess.run(model.mean_detector.output, {model.input_data: train_inputs_batch, 
                                      model.summary_length: train_outputs_lengths, 
                                      model.text_length: train_inputs_lengths,
                                      model.keep_prob: 1.0, model.min_mean: data.min_mean, 
                                      model.max_mean: data.max_mean})
    
    test_predictions = sess.run(model.mean_detector.output, {model.input_data: inputs_batch, 
                                      model.summary_length: outputs_lengths, 
                                      model.text_length: inputs_lengths,
                                      model.keep_prob: 1.0, model.min_mean: data.min_mean, 
                                      model.max_mean: data.max_mean})


fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(range(start,end), train_inputs_batch[0][start:end])
ax2.plot(range(start,end), train_outputs_batch[0][start:end], '--', range(start,end), train_predictions[0][start:end], '-r')
fig.savefig("train_mean.png", bbox_inches='tight')
#plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(range(start,end), inputs_batch[0][start:end])
ax2.plot(range(start,end), outputs_batch[0][start:end], '--', range(start,end), test_predictions[0][start:end], '-r')
fig.savefig("test_mean.png", bbox_inches='tight')
#plt.show()
logging.info("Run complete.")


# In[ ]:



