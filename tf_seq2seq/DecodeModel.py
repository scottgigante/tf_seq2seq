import tensorflow as tf
from tensorflow.python.layers.core import Dense

from Optimizer import Optimizer

class DecodeModel:
    
    def __init__(self, inputs, input_length, targets, target_length, max_target_length, 
                 num_layers, rnn_size, sample_prob, keep_prob, learning_rate, num_labels, 
                 embedding_size, batch_size, length_cost_prop, GO, EOS):
        self.GO = GO
        self.EOS = EOS
        self.batch_size = batch_size
        self.num_labels = num_labels
        
        with tf.name_scope("decoder") as scope:
            self.output_embeddings = self.generate_embeddings(embedding_size)
            self.length_predictions = self.length_detection(inputs, target_length)
            self.logits, self.predictions = self.decoding_layer(inputs, input_length, 
                                       targets, target_length, 
                                       max_target_length, num_layers, rnn_size, sample_prob, keep_prob)
            self.cost = self.calculate_loss(self.logits, targets, self.length_predictions, 
                                    target_length, max_target_length, length_cost_prop)
            self.optimizer = Optimizer(learning_rate)
            self.train_op = self.optimizer.apply_gradients(self.cost)
        self.summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='decoder'))

    def generate_embeddings(self, embedding_size):
        embeddings = tf.Variable(tf.truncated_normal(
                shape=[self.num_labels, embedding_size], dtype=tf.float32), 
            name="output_embeddings")
        tf.summary.histogram('embedding', embeddings)
        return embeddings
    
    def process_encoding_input(self, targets):
        '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
        ending = tf.strided_slice(targets, [0, 0], [self.batch_size, -1], [1, 1], name="target_ending")
        dec_input = tf.concat([tf.fill([self.batch_size, 1], self.GO), ending], 1, name="decoder_input")
        dec_embed_input = tf.nn.embedding_lookup(self.output_embeddings, dec_input, name="decoder_embedded_input")
        
        return dec_embed_input
    
    def length_detection(self, inputs, target_length):
        
        with tf.name_scope("length"):
            dense = tf.layers.dense(inputs,
                                    1,
                                    activation=tf.nn.relu)
            prediction = tf.cast(tf.round(dense), tf.int32)
        
        return prediction
    
    def decoding_layer(self, inputs, input_length, targets, target_length, max_target_length, num_layers, rnn_size, sample_prob, keep_prob):
    
        with tf.name_scope("hidden"):
            dec_input = self.process_encoding_input(targets)
        
            lstm_cells = []
            for layer in range(num_layers):
                with tf.variable_scope('decoder_{}'.format(layer)):
                    lstm = tf.contrib.rnn.LSTMCell(rnn_size,
                          initializer=tf.random_uniform_initializer(-0.1, 0.1))
                    dropout = tf.contrib.rnn.DropoutWrapper(
                                   lstm, 
                                   input_keep_prob = keep_prob)
                    lstm_cells.append(dropout)
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)

            output_layer = Dense(self.num_labels,
                   kernel_initializer = tf.truncated_normal_initializer(  
                                            mean=0.0, 
                                            stddev=0.1),
                   name="dense")

            attn_mech = tf.contrib.seq2seq.BahdanauAttention(
                              rnn_size,
                              inputs,
                              normalize=False,
                              name='BahdanauAttention')
            attn_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(stacked_lstm,
                                                           attn_mech,
                                                           rnn_size,
                                                           name="attention_cell")
        
            initial_state = attn_cell.zero_state(self.batch_size, tf.float32)
            
        
        with tf.variable_scope("output"):
            training_logits = self.training_decoding_layer(
                                  attn_cell, 
                                  initial_state,
                                  output_layer,
                                  dec_input, 
                                  target_length, 
                                  max_target_length,
                                  sample_prob)
            logits = tf.identity(training_logits.rnn_output, name='logits')

        with tf.variable_scope("output", reuse=True):
            inference_logits = self.inference_decoding_layer(
                                  attn_cell, 
                                  initial_state, 
                                  output_layer)
            predictions = tf.identity(inference_logits.sample_id, name='predictions')
            
        return logits, predictions
    
    def inference_decoding_layer(self, attn_cell, initial_state, output_layer):
    
        start_tokens = tf.tile(tf.constant([self.GO],  
                                           dtype=tf.int32),  
                                           [self.batch_size], 
                                           name='start_tokens')
    
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(  
                               self.output_embeddings,
                               start_tokens,
                               self.EOS)
                
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                                attn_cell,
                                inference_helper,
                                initial_state,
                                output_layer)
                
        inference_logits, _ = tf.contrib.seq2seq.dynamic_decode(
                                inference_decoder,
                                output_time_major=False,
                                impute_finished=True,
                                maximum_iterations=None)#tf.reduce_max(self.length_predictions))
        return inference_logits
        
    def training_decoding_layer(self, attn_cell, initial_state, output_layer, 
                                dec_input, target_length, max_target_length,
                                sample_prob):

        #training_helper = tf.contrib.seq2seq.TrainingHelper( 
        #                     inputs=dec_input,
        #                     sequence_length=target_length,
        #                     time_major=False)
        training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                                inputs=dec_input,
                                sequence_length=target_length,
                                embedding=self.output_embeddings,
                                sampling_probability=sample_prob,
                                time_major=False)     
                             
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
                              attn_cell,
                              training_helper,
                              initial_state,
                              output_layer)
                              
        training_logits, _ = tf.contrib.seq2seq.dynamic_decode(
                                training_decoder,
                                output_time_major=False,
                                impute_finished=True,
                                maximum_iterations=None)#max_target_length)
        return training_logits
    
    def calculate_loss(self, predictions, targets, length_predictions, target_length, max_target_length, length_cost_prop):
        with tf.name_scope("loss"):
            masks = tf.sequence_mask(target_length, max_target_length, name='sequence_masks')
            shifted_masks = tf.concat([tf.strided_slice(masks, [0, 1], [self.batch_size, max_target_length], [1, 1], name="masks_ending"), tf.fill([self.batch_size,1], False)], 1, name='shifted_masks')
            self.weights = weights = tf.add(tf.cast(masks, tf.float32), tf.cast(tf.logical_xor(masks, shifted_masks), tf.float32) * 10., name="weights")
            seq_cost = tf.contrib.seq2seq.sequence_loss(
                    predictions,
                    targets,
                    weights,
                    name="sequence_loss")
                    
            scaled_lengths = target_length / max_target_length
            scaled_length_preds = length_predictions / max_target_length
            length_cost = tf.cast(tf.reduce_mean(tf.abs(target_length - length_predictions) / target_length, name="length_loss"), tf.float32)
            
            cost = length_cost_prop * length_cost + (1-length_cost_prop) * seq_cost
            
        tf.summary.scalar("seq_loss", seq_cost)
        tf.summary.scalar("length_loss", length_cost)
        tf.summary.scalar("loss", cost)
        
        return cost
