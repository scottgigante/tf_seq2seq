import numpy as np

import multiprocessing

from DataContainer import DataContainer

class FakeFast5Loader(object):
    
    def __init__(self, num_features, min_event_length, max_event_length, num_classes,
            VOCAB_LOWER, model):
        self.num_features = num_features
        self.min_event_length = min_event_length
        self.max_event_length = max_event_length
        self.num_classes = num_classes
        self.VOCAB_LOWER = VOCAB_LOWER
        self.model = model
    
    def generateRead(self, i):
        # Should be a better way of running map without actually needing an argument
        # Generating random input
        example_inputs = []
        example_labels = []
        example_edges = []
        example_means = []
        previous_mu = 0
        while len(example_inputs) < self.num_features:
            event_length = min(np.random.randint(self.min_event_length, self.max_event_length), 
                               self.num_features - len(example_inputs))
            label = np.random.randint(self.num_classes) + self.VOCAB_LOWER
            mu, sigma = self.model[label - self.VOCAB_LOWER]
            event = [max(self.VOCAB_LOWER,int(i)) for i in np.random.normal(mu, sigma, event_length)]
            example_inputs.extend(event)
            example_labels.append(label)
            
            example_means.extend([np.mean(event)] * event_length)
            example_edges.extend([0] * event_length)
            if previous_mu < mu:
                example_edges[-event_length] = 1
            elif previous_mu > mu:
                example_edges[-event_length] = -1
            previous_mu = mu
        
        return example_inputs, example_labels, example_edges, example_means

class FakeNanoporeData(DataContainer):
    
    def __init__(self,
                 batch_size,
                 num_examples,
                 num_features = 1024,
                 num_classes = 4,
                 min_event_length = 4,
                 max_event_length = 20,
                 mu_mean = 360,
                 mu_sd = 25,
                 sigma_mean = 11,
                 sigma_shape = 3,
                 val_proportion = 0.05,
                 test_proportion = 0.05):
        
        self.min_event_length = min_event_length
        self.max_event_length = max_event_length
        self.mu_mean = mu_mean
        self.mu_sd = mu_sd
        self.sigma_mean = sigma_mean
        self.sigma_shape = sigma_shape
        
        self.model = self.generate_model(num_classes)
        
        super().__init__(batch_size,
                 num_examples,
                 num_features,
                 num_classes,
                 val_proportion,
                 test_proportion)
    
    def generate_model(self, num_classes):
        model = {}
        for class_i in range(num_classes):
            mu = np.random.normal(self.mu_mean, self.mu_sd, 1)
            sigma = np.random.wald(self.sigma_mean, self.sigma_shape, 1)
            model[class_i] = (mu, sigma)
        return model
    
    def generate_data(self, start, end):
        num_examples = end-start
        # num_examples should be a multiple of self.batch_size
        if num_examples % self.batch_size > 0:
            num_examples += self.batch_size - num_examples % self.batch_size
        inputs = []
        labels = []
        edges = []
        means=[]
        
        loader = FakeFast5Loader(self.num_features, self.min_event_length, 
                self.max_event_length, self.num_classes,
                self.VOCAB_LOWER, self.model)
        pool = multiprocessing.Pool()
        for example_inputs, example_labels, example_edges, example_means in pool.imap_unordered(loader.generateRead, range(num_examples)):
            #example_inputs = normalise_inputs(example_inputs)
            inputs.append(example_inputs)
            labels.append(example_labels)
            edges.append(example_edges)
            means.append(example_means)
        pool.close()
        pool.join()
            
        return inputs, labels, edges, means
