import tensorflow as tf

class EncodeModel:
    
    def __init__(self, input_data, input_embeddings, num_layers, conv_size, conv_widths, keep_prob, output_size):
        self.output = self.encoding_layer(input_data, input_embeddings, num_layers, conv_size, conv_widths, keep_prob, output_size)

    def encoding_layer(self, input_data, input_embeddings, num_layers, conv_size, conv_widths, keep_prob, output_size):
        with tf.name_scope("encoder") as scope:
            enc_embed_input = tf.nn.embedding_lookup(input_embeddings, input_data, name="encode_embedding")
        
            conv_outputs = []
            out_channels = conv_size
    
            for width in conv_widths:
                inputs = enc_embed_input
                in_channels = 1
                for layer in range(num_layers):
                    with tf.variable_scope('encoder_{}_{}'.format(layer, width)):
                        filter = tf.Variable(tf.truncated_normal([width, in_channels, out_channels]), name="filter")
                        conv = tf.nn.conv1d(inputs, filter, stride=1, padding="SAME", name="conv_output")
                        dropout = tf.layers.dropout(inputs=conv, rate=keep_prob, name="conv_output_dropout")
                    inputs = dropout
                    in_channels = out_channels
                conv_outputs.append(dropout)
    
            concat_conv = tf.concat(conv_outputs,axis=2, name="concat_conv")
    
            enc_output = tf.layers.dense(inputs=concat_conv, units=output_size, activation=tf.nn.relu, name="enc_output")
            dropout = tf.layers.dropout(inputs=enc_output, rate=keep_prob, name="enc_output_dropout")
    
        return dropout
