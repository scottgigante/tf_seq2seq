import numpy as np

import glob
import os
import h5py
import multiprocessing
import functools
import logging
from copy import deepcopy

from DataContainer import DataContainer

class Fast5Loader(object):
    # scaling factor taken from relationship between raw and event data
    __scale = 5.4
    # max and min values from nanopolish model:
    # min(level_mean - 2*level_sd - 2*(sd_mean + 2*sd_sd))
    # max(level_mean + 2*level_sd + 2*(sd_mean + 2*sd_sd))
    __max = 142.21 * __scale
    __min = 40.26 * __scale
    __median = 432
    __std = 70
    __events_path = "Analyses/AlignToRef/CurrentSpaceMapped_template/Events"
    __labels = {
        'A' : 0,
        'C' : 1,
        'G' : 2,
        'T' : 3,
    }
    
    def __init__(self, vocab_lower, max_length, EOS):
        self.max_length = max_length
        self.labels = deepcopy(self.__labels)
        self.EOS = EOS
        for key in self.labels:
            self.labels[key] += vocab_lower

    def clip_spikes(self, inputs):
        clip = np.where(np.logical_or(inputs > self.__max, inputs < self.__min))[0]
        left, right = -1, -1
        for i in clip:
            if i < right:
                # already clipped
                continue
            left, right = i-1, i+1
            while left in clip:
                left -= 1
            while right in clip:
                right += 1
            if left < 0:
                # clip everything from start
                inputs[:right] = inputs[right]
            elif right >= len(inputs):
                # clip everything to end
                inputs[left+1:] = inputs[left]
            else:
                inputs[left+1:right] = (inputs[left] + inputs[right]) / 2
            
        return inputs
    
    def median_normalize(self, inputs):
        inputs = self.clip_spikes(inputs)
        inputs = inputs - np.median(inputs)
        inputs = inputs / np.std(inputs) * self.__std
        inputs = inputs + self.__median
        return np.array(np.round(inputs), dtype=np.int32)
    
    def load_data(self, filename):
        inputs = []
        labels = []
        edges = []
        means = []
        try:
            with h5py.File(filename) as h5file:
                file_inputs = h5file["Raw/Reads"][[k for k in h5file["Raw/Reads"].keys()][0]]["Signal"][()]
                file_inputs = self.median_normalize(file_inputs)
                start_time = h5file["Raw/Reads"][[k for k in h5file["Raw/Reads"].keys()][0]].attrs['start_time']
                sampling_rate = h5file["UniqueGlobalKey/channel_id"].attrs["sampling_rate"]
                length = np.array(h5file[self.__events_path][()]["length"] * sampling_rate,
                                  dtype=np.int32)
                start = np.array(h5file[self.__events_path][()]["start"] * sampling_rate, 
                                 dtype=np.int32) - start_time
                kmer = [k.decode("utf-8")[2] for k in h5file[self.__events_path][()]["kmer"]]
                end = start + length
                example_inputs = []
                example_labels = []
                example_edges = []
                example_means = []
                previous_mu = 0
                trim = (len(file_inputs) % self.max_length) // 2
                for i in range(len(start)):
                    # check start
                    if start[i] < trim:
                        continue
                    # check length
                    if len(example_inputs) > self.max_length:
                        # add to output
                        example_labels.append(self.EOS)
                        inputs.append(np.array(example_inputs))
                        labels.append(np.array(example_labels))
                        edges.append(np.array(example_edges))
                        means.append(np.array(example_means))
                        example_inputs = []
                        example_labels = []
                        example_edges = []
                        example_means = []
                    # for each event
                    event_length = end[i] - start[i]
                    if event_length <= 1:
                        continue
                    example_inputs.extend(file_inputs[start[i]:end[i]])
                    mu = np.mean(file_inputs[start[i]:end[i]])
                    example_means.extend([mu] *  (event_length))
                    example_edges.extend([0] * (event_length))
                    if previous_mu < mu:
                        example_edges[-event_length] = 1
                    elif previous_mu > mu:
                        example_edges[-event_length] = -1
                    example_labels.append(self.labels[kmer[i]])
                    previous_mu = mu
                    
        except OSError as e:
            print(str(e))
            return [], [], [], []
        except Exception as e:
            print("Caught exception in worker thread")
            print(str(e))
            raise
        
        return inputs, labels, edges, means

class NanoporeData(DataContainer):
    
    #__events_path = "Analyses/Basecall_1D_000/BaseCalled_template/Events"
    def __init__(self,
                 data_dir,
                 batch_size,
                 max_files = None,
                 num_classes = 4,
                 num_features = 1024,
                 val_proportion = 0.1,
                 test_proportion = 0.1):
        
        self.data_dir = data_dir
        self.file_list = self.load_files(data_dir)
        if max_files is not None:
            self.file_list = np.random.choice(self.file_list, size=(max_files), replace=False)
            
        total_examples = len(self.file_list)
        
        super().__init__(batch_size,
                 total_examples,
                 num_features,
                 num_classes,
                 val_proportion,
                 test_proportion)
    
    def load_files(self, data_dir):
        return glob.glob(os.path.join(data_dir, "*.fast5"))
    
    def generate_data(self, start, end, max_length=1024):
        file_list = self.file_list[start:end]
        file_loader = Fast5Loader(self.VOCAB_LOWER, max_length, self.EOS)
        inputs = []
        labels = []
        edges = []
        means = []
        pool = multiprocessing.Pool()
        i = 0
        for file_inputs, file_labels, file_edges, file_means in pool.imap_unordered(file_loader.load_data, file_list):
            i += 1
            inputs.extend(file_inputs)
            labels.extend(file_labels)
            edges.extend(file_edges)
            means.extend(file_means)
        pool.close()
        pool.join()
        
        if len(inputs) < self.batch_size:
            raise Exception("Insufficient data: samples ({}) less than batch size ({})".format(len(inputs), self.batch_size))
        
        return inputs, labels, edges, means
