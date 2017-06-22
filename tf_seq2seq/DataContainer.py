import numpy as np

import math
import logging

class DataContainer(object):
    
    def __init__(self,
                 batch_size,
                 total_examples,
                 num_features = 1024,
                 num_classes = 4,
                 val_proportion = 0.05,
                 test_proportion = 0.05):
        
        val_examples = round(total_examples * val_proportion)
        test_examples = round(total_examples * test_proportion)
        num_examples = total_examples - val_examples - test_examples
        
        self.batch_size = batch_size
        self.num_features = num_features
        self.num_classes = num_classes
        
        self.PAD = 0
        self.GO = 1
        self.EOS = 2
        self.VOCAB_LOWER = 3
        
        logging.info("Generating training data...")
        self.inputs, self.labels, self.edges, self.means = self.generate_data(0, num_examples)
        logging.info("{} examples generated.".format(len(self.inputs)))
        logging.info("Generating validation data...")
        self.val_inputs, self.val_labels, self.val_edges, self.val_means = self.generate_data(num_examples, num_examples + val_examples)
        logging.info("{} examples generated.".format(len(self.val_inputs)))
        logging.info("Generating test data...")
        self.test_inputs, self.test_labels, self.test_edges, self.test_means = self.generate_data(num_examples + val_examples, num_examples + val_examples + test_examples)
        logging.info("{} examples generated.".format(len(self.test_inputs)))
        
        self.num_examples = len(self.inputs)
        self.input_embedding_matrix = self.generate_input_embeddings(self.inputs + self.val_inputs + self.test_inputs)  
        self.min_mean, self.max_mean = self.get_range(self.means + self.val_means + self.test_means)
    
    def generate_data(self, start, end):
        raise NotImplementedError("Please Implement this method")
        return inputs, labels, edges, means
    
    def generate_input_embeddings(self, in_list):
        in_vocab = [item for sublist in in_list for item in sublist]
        in_vocab_size = max(in_vocab)+1
        in_vocab_mean = np.mean(in_vocab)
        in_vocab_std = np.std(in_vocab)
        
        return np.asarray([[(i-in_vocab_mean)/in_vocab_std] for i in range(in_vocab_size)], dtype=np.float32)
    
    def embedding_lookup(self, x):
        return [np.reshape(self.input_embedding_matrix[np.asarray(i, dtype=np.uint32)], -1) for i in x]
    
    def pad_batch(self, batch):
        """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
        max_sentence = max([len(sentence) for sentence in batch])
        return [np.concatenate([sentence, [self.PAD] * (max_sentence - len(sentence))]) if len(sentence) < max_sentence else sentence for sentence in batch]
    
    def get_batch(self, inputs, labels, start):
        outputs_batch = labels[start:start + self.batch_size]
        inputs_batch = inputs[start:start + self.batch_size]
        pad_outputs_batch = np.array(self.pad_batch(  
                                  outputs_batch))
        pad_inputs_batch = np.array(self.pad_batch(inputs_batch))

        outputs_lengths = []
        for output in outputs_batch:
            outputs_lengths.append(len(output))

        inputs_lengths = []
        for inp in inputs_batch:
            inputs_lengths.append(len(inp))
        return (pad_outputs_batch, 
                pad_inputs_batch, 
                outputs_lengths, 
                inputs_lengths)
    
    def get_batches(self, inputs, labels):
        inputs, labels = self.shuffle_sequences(inputs, labels)
        num_batches = len(inputs) // self.batch_size
        for batch_i in range(0, num_batches):
            start_i = batch_i * self.batch_size
            yield self.get_batch(inputs, labels, start_i)
    
    def shuffle_sequences(self, inputs, labels):
        # shuffle inputs so next epoch is not the same
        indices = list(range(len(inputs)))
        np.random.shuffle(indices)
        inputs = list(map(inputs.__getitem__, indices))
        labels = list(map(labels.__getitem__, indices))
        return inputs, labels
    
    def get_train_batches(self, label_edges=False, label_means=False):
        if label_edges:
            return self.get_batches(self.inputs, self.edges)
        elif label_means:
            return self.get_batches(self.inputs, self.means)
        else:
            return self.get_batches(self.inputs, self.labels)
    
    def get_val_batches(self, label_edges=False, label_means=False):
        if label_edges:
            return self.get_batches(self.val_inputs, self.val_edges)
        elif label_means:
            return self.get_batches(self.val_inputs, self.val_means)
        else:
            return self.get_batches(self.val_inputs, self.val_labels)
    
    def get_test_batches(self, label_edges=False, label_means=False):
        if label_edges:
            return self.get_batches(self.test_inputs, self.test_edges)
        elif label_means:
            return self.get_batches(self.test_inputs, self.test_means)
        else:
            return self.get_batches(self.test_inputs, self.test_labels)
    
    def num_batches(self):
        return self.num_examples // self.batch_size
    
    def get_input_vocab_freq(self):
        in_vocab = [item for sublist in self.inputs for item in sublist]
        vocab = {}
        for item in in_vocab:
            try:
                vocab[item] += 1
            except KeyError:
                vocab[item] = 1
        return vocab
    
    def get_range(self, in_list):
        means = [item for sublist in in_list for item in sublist]
        return math.floor(min(means)), math.ceil(max(means))
