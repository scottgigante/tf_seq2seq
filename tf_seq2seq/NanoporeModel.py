import tensorflow as tf
import numpy as np

from EncodeModel import EncodeModel
from DecodeModel import DecodeModel
from EdgeDetectModel import EdgeDetectModel
from MeanDetectModel import MeanDetectModel

class NanoporeModel:
    
    def __init__(self,
                 input_embeddings,
                 num_classes,
                 batch_size,
                 output_embedding_size = 16,
                 conv_size = 64,
                 conv_widths = [4, 10, 20],
                 num_encode_layers = 1,
                 rnn_size = 64,
                 num_decode_layers = 3,
                 attention=True,
                 seed=42):
        self.input_embeddings = input_embeddings
        self.batch_size = batch_size
        self.output_embedding_size = output_embedding_size
        self.conv_size = conv_size
        self.conv_widths = conv_widths
        self.num_encode_layers = num_encode_layers
        self.rnn_size = rnn_size
        self.num_decode_layers = num_decode_layers
        self.attention = attention
        self.seed = seed
        
        self.PAD = 0
        self.GO = 1
        self.EOS = 2
        self.VOCAB_LOWER = 3
        
        self.num_labels = num_classes + self.VOCAB_LOWER
        
        self.train_graph = self.build_model_graph()
        
    def init_model_inputs(self):
        with tf.name_scope("input_data"):
            self.input_data = tf.placeholder(tf.int32,[self.batch_size, None], name='input')
            self.targets = tf.placeholder(tf.int32, [self.batch_size, None], name='targets')
            self.min_mean = tf.placeholder(tf.int32, name="min_mean")
            self.max_mean = tf.placeholder(tf.int32, name="max_mean")
            
            self.text_length = tf.placeholder(tf.int32, (None,),       
                                         name='text_length')
            self.summary_length = tf.placeholder(tf.int32, (None,), 
                                            name='summary_length')
            self.max_summary_length = tf.reduce_max(self.summary_length, 
                                               name='max_dec_len')
            
        with tf.name_scope("learn_parameters"):
            self.lr = tf.placeholder(tf.float32, (), name='learning_rate')
            self.keep_prob = tf.placeholder(tf.float32, (), name='keep_prob')
            self.sample_prob = tf.placeholder(tf.float32, (), name='sample_prob')
            self.length_cost_prop = tf.placeholder(tf.float32, (), name='length_cost_prop')
    
    def build_model_graph(self):
        
        # Build the graph
        tf.reset_default_graph()
        train_graph = tf.Graph()
        # Set the graph to default to ensure that it is ready for training
        with train_graph.as_default():
            
            # Initialize model inputs   
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)
            self.init_model_inputs()
            
            # Build the graph.
            self.encoder = EncodeModel(self.input_data, self.input_embeddings, self.num_encode_layers, self.conv_size, self.conv_widths, self.keep_prob, self.output_embedding_size)
            self.edge_detector = EdgeDetectModel(self.encoder.output, self.targets, self.lr, self.batch_size)
            self.mean_detector = MeanDetectModel(self.encoder.output, self.targets, self.lr, self.min_mean, self.max_mean)
            self.decoder = DecodeModel(self.encoder.output, self.text_length, 
                    self.targets, self.summary_length, self.max_summary_length, 
                    self.num_decode_layers, self.rnn_size, self.sample_prob, self.keep_prob, self.lr, 
                    self.num_labels, self.output_embedding_size, self.batch_size, 
                    self.length_cost_prop, self.GO, self.EOS)
            
            self.saver = tf.train.Saver() 
            
        return train_graph
    
    def rebuild_model_graph(self, params):
        for key, value in params.items():
            if value is not None:
                setattr(self, key, value)
        self.train_graph = self.build_model_graph()
