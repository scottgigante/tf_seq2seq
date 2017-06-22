import tensorflow as tf

import os
import time

class NanoporeSeq2SeqTrainer(object):
    
    def __init__(self,
                 model,
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
                 checkpoint="best_model.ckpt",
                 verbose=True, 
                 save_best=True,
                 display_per_epoch=2,
                 update_per_epoch=1,
                 log_dir="seq2seq_log"):
        self.model = model
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.sample_prob = sample_prob
        self.sample_prob_decay = sample_prob_decay
        self.min_sample_prob = min_sample_prob
        self.keep_probability = keep_probability
        self.length_cost_prop = length_cost_prop
        self.epochs = epochs
        self.max_tests_no_best = max_tests_no_best 
        self.checkpoint = checkpoint
        self.verbose=verbose
        self.save_best=save_best
        self.display_per_epoch=display_per_epoch
        self.update_per_epoch=update_per_epoch
        self.log_dir = log_dir
    
    def train_model(self, sess, data, model, epochs, num_batches, display_step, update_step, 
                    max_tests_no_best, learning_rate, min_learning_rate, learning_rate_decay, sample_prob, sample_prob_decay, min_sample_prob, keep_probability,
                    length_cost_prop, verbose, save_best, writers, checkpoint, label_edges=False, label_means=False, label_seq=False):
        if label_edges:
            train_op = model.edge_detector.train_op
            cost = model.edge_detector.cost
            summary_op = model.edge_detector.summary_op
        elif label_means:
            train_op = model.mean_detector.train_op
            cost = model.mean_detector.cost
            summary_op = model.mean_detector.summary_op
        elif label_seq:
            train_op = model.decoder.train_op
            cost = model.decoder.cost
            summary_op = model.decoder.summary_op
        else:
            raise Exception("No labelling method found")
            
        update_loss = 0
        batch_loss = 0
        tests_no_best = 0
        summary_update_loss = [] # Record the update losses for saving improvements in the model
        try:
            for epoch_i in range(1, epochs+1):
                for batch_i, (outputs_batch, inputs_batch, outputs_lengths, inputs_lengths) in enumerate(
                        data.get_train_batches(label_edges=label_edges, label_means=label_means)):
                    start_time = time.time()
                    _, loss, summary = sess.run(
                        [train_op, cost, summary_op],
                        {model.input_data: inputs_batch,
                         model.targets: outputs_batch,
                         model.summary_length: outputs_lengths,
                         model.text_length: inputs_lengths,
                         model.lr: learning_rate,
                         model.sample_prob: sample_prob,
                         model.keep_prob: keep_probability,
                         model.length_cost_prop: length_cost_prop,
                         model.min_mean: data.min_mean,
                         model.max_mean: data.max_mean})

                    batch_loss += loss
                    end_time = time.time()
                    batch_time = end_time - start_time

                    if batch_i % display_step == 0 and (epoch_i > 1 or batch_i > 0):
                        writers['train'].add_summary(summary, epoch_i * num_batches + batch_i)
                        #print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                        val_outputs, val_inputs, val_outputs_lengths, val_inputs_lengths = next(data.get_val_batches(label_edges=label_edges, label_means=label_means))
                        test_outputs, test_inputs, test_outputs_lengths, test_inputs_lengths = next(data.get_test_batches(label_edges=label_edges, label_means=label_means))
                        val_loss, summary = sess.run(
                        [cost, summary_op],
                        {model.input_data: val_inputs,
                         model.targets: val_outputs,
                         model.summary_length: val_outputs_lengths,
                         model.text_length: val_inputs_lengths,
                         model.sample_prob: sample_prob,
                         model.keep_prob: 1.0,
                         model.length_cost_prop: length_cost_prop,
                         model.min_mean: data.min_mean,
                         model.max_mean: data.max_mean})
                        writers['val'].add_summary(summary, epoch_i * num_batches + batch_i)
                        test_loss, summary = sess.run(
                        [cost, summary_op],
                        {model.input_data: test_inputs,
                         model.targets: test_outputs,
                         model.summary_length: test_outputs_lengths,
                         model.text_length: test_inputs_lengths,
                         model.sample_prob: sample_prob,
                         model.keep_prob: 1.0,
                         model.length_cost_prop: length_cost_prop,
                         model.min_mean: data.min_mean,
                         model.max_mean: data.max_mean})
                        writers['test'].add_summary(summary, epoch_i * num_batches + batch_i)
                        
                        if verbose:
                            print('{:{}d}/{}\t{:{}d}/{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.2f}\t'
                              .format(epoch_i,
                                      len(str(self.epochs)),
                                      self.epochs, 
                                      batch_i, 
                                      len(str(num_batches)), 
                                      num_batches, 
                                      batch_loss / display_step, 
                                      val_loss,
                                      test_loss,
                                      batch_time*display_step), end='')

                        batch_loss = 0

                        if batch_i % update_step == 0:
                            if sample_prob < min_sample_prob:
                                print(sample_prob, min_sample_prob)
                            else:
                                #if label_means and batch_i == 0 and epoch_i == 12:
                                #    tmp1, tmp2, tmp3, tmp4 = data.test_inputs, data.test_labels, data.test_edges, data.test_means
                                #    data.test_inputs, data.test_labels, data.test_edges, data.test_means = data.val_inputs, data.val_labels, data.val_edges, data.val_means
                                #    data.val_inputs, data.val_labels, data.val_edges, data.val_means = tmp1, tmp2, tmp3, tmp4
                                #print("Average loss for this update:", round(update_loss/update_check,3))
                                summary_update_loss.append(val_loss)

                                # If the update loss is at a new minimum, save the model
                                if val_loss <= min(summary_update_loss):
                                    if verbose:
                                        print('yes') 
                                    tests_no_best = 0
                                    if save_best:
                                        model.saver.save(sess, checkpoint)

                                else:
                                    if verbose:
                                        print("no")
                                    tests_no_best += 1
                                    if tests_no_best == max_tests_no_best:
                                        break
                        else:
                            if verbose:
                                print()
                    
                # TODO: decay sample prob once per epoch, not once per batch
                sample_prob = 1 - (1-sample_prob) * sample_prob_decay
                print(sample_prob)
                if sample_prob > 1:
                    sample_prob = 1

                # Reduce learning rate, but not below its minimum value
                learning_rate *= learning_rate_decay
                if learning_rate < min_learning_rate:
                    learning_rate = min_learning_rate

                if tests_no_best == max_tests_no_best:
                    break
        except KeyboardInterrupt:
            if len(summary_update_loss) > 0:
                pass
            else:
                raise
        #if label_means and epoch_i >= 12:
        #    tmp1, tmp2, tmp3, tmp4 = data.test_inputs, data.test_labels, data.test_edges, data.test_means
        #    data.test_inputs, data.test_labels, data.test_edges, data.test_means = data.val_inputs, data.val_labels, data.val_edges, data.val_means
        #    data.val_inputs, data.val_labels, data.val_edges, data.val_means = tmp1, tmp2, tmp3, tmp4
        print("Stopping Training.")
        return min(summary_update_loss)
    
    def run_model_training(self,
                           data, 
                           learning_rate=None,
                           min_learning_rate=None,
                           learning_rate_decay=None,
                           sample_prob=None, 
                           sample_prob_decay=None,
                           min_sample_prob=None,
                           keep_probability=None,
                           length_cost_prop=None,
                           epochs=None,
                           max_tests_no_best=None,
                           checkpoint=None,
                           verbose=None, 
                           save_best=None,
                           display_per_epoch=None,
                           update_per_epoch=None,
                           log_dir = None,
                           restore=False,
                           log_subdir=None):
                           
        learning_rate = learning_rate or self.learning_rate
        min_learning_rate = min_learning_rate or self.min_learning_rate
        learning_rate_decay = learning_rate_decay or self.learning_rate_decay
        sample_prob = sample_prob or self.sample_prob
        sample_prob_decay = sample_prob_decay or self.sample_prob_decay
        min_sample_prob = min_sample_prob or self.min_sample_prob
        keep_probability = keep_probability or self.keep_probability
        length_cost_prop = length_cost_prop or self.length_cost_prop
        epochs = epochs or self.epochs
        max_tests_no_best = max_tests_no_best or self.max_tests_no_best
        checkpoint = checkpoint or self.checkpoint
        verbose = verbose or self.verbose
        save_best = save_best or self.save_best
        display_per_epoch = display_per_epoch or self.display_per_epoch
        update_per_epoch = update_per_epoch or self.update_per_epoch
        log_dir = log_dir or self.log_dir
        model = self.model
        
        i=0
        while any(x.startswith(str(i)) for x in os.listdir(log_dir)):
            i += 1
        log_dir = os.path.join(log_dir, str(i))
        if log_subdir is not None:
            log_dir = "{}_{}".format(log_dir, log_subdir)
        
        num_batches = data.num_batches()
        display_step = max(num_batches // display_per_epoch, 1)
        update_step = max(num_batches // update_per_epoch, 1)
        
        writers = {
            "train" : tf.summary.FileWriter(log_dir+"_train", graph=model.train_graph),
            "val" : tf.summary.FileWriter(log_dir+"_val", graph=model.train_graph),
            "test" : tf.summary.FileWriter(log_dir+"_test", graph=model.train_graph),
        }
    
        if verbose:
            print("Epoch\tBatch\t\tLoss\tVal\tTest\tTime\tNew Best")
        with tf.Session(graph=model.train_graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            if False:
                # Run edge and mean detection
                # If we want to continue training a previous session
                if not restore:
                    edge_loss = self.train_model(sess, data, model, epochs, num_batches, 
                            display_step, update_step,
                            max_tests_no_best, learning_rate, min_learning_rate, 
                            learning_rate_decay, sample_prob, sample_prob_decay, 
                            min_sample_prob, keep_probability, length_cost_prop,
                            verbose, save_best, writers, "edge_" + checkpoint, 
                            label_edges=True)
                #model.saver.restore(sess, "edge_" + checkpoint)
            
                mean_loss = self.train_model(sess, data, model, epochs, num_batches, 
                        display_step, update_step,  
                        max_tests_no_best, learning_rate, min_learning_rate, 
                        learning_rate_decay, sample_prob, sample_prob_decay, 
                        min_sample_prob, keep_probability, length_cost_prop,
                        verbose, save_best, writers, "mean_" + checkpoint, 
                        label_means=True)
            seq_loss = self.train_model(sess, data, model, epochs, num_batches, 
                        display_step, update_step, 
                        max_tests_no_best, learning_rate, min_learning_rate, 
                        learning_rate_decay, sample_prob, sample_prob_decay, 
                        min_sample_prob, keep_probability, length_cost_prop,
                        verbose, save_best, writers, "seq_" + checkpoint, label_seq=True)

        return seq_loss
    
    def print_dict(self, d):
        return '_'.join(["{}={}".format(key, value) for key, value in d.items() if value is not None])
    
    def test_parameters(self, data, model_kwargs={}, learn_kwargs={}):
        log_subdir = self.print_dict(model_kwargs) + '_' + self.print_dict(learn_kwargs)
        if model_kwargs:
            self.model.rebuild_model_graph(model_kwargs)
        return self.run_model_training(data, **learn_kwargs, log_subdir=log_subdir)
    
    def parameter_search(self, data, model_params, learn_params, model_kwargs={}, learn_kwargs={}):
        """
        params: a dictionary of parameters to be passed to run_model_training, each as a 
        list of candidate parameters
        """
        if len(model_params) + len(learn_params) == 0:
            # we've selected a config. test!
            return self.test_parameters(data, model_kwargs, learn_kwargs), model_kwargs, learn_kwargs
        else:
            # recursive step: select another parameter.
            min_loss = None
            if len(model_params) > 0:
                # start with model params
                param, values = next(iter(model_params.items()))
                new_params = model_params.copy()
                del new_params[param]
                new_kwargs = model_kwargs.copy()
                for value in values:
                    new_kwargs[param] = value
                    loss, model, learn = self.parameter_search(
                            data, new_params, learn_params, new_kwargs, learn_kwargs)
                    if min_loss is None or loss < min_loss:
                        min_loss = loss
                        min_model_kwargs = model
                        min_learn_kwargs = learn
            else:
                # no model params, start with learn
                param, values = next(iter(learn_params.items()))
                new_params = learn_params.copy()
                del new_params[param]
                new_kwargs = learn_kwargs.copy()
                for value in values:
                    new_kwargs[param] = value
                    loss, model, learn = self.parameter_search(
                            data, model_params, new_params, model_kwargs, new_kwargs)
                    if min_loss is None or loss < min_loss:
                        min_loss = loss
                        min_model_kwargs = model
                        min_learn_kwargs = learn
            return min_loss, min_model_kwargs, min_learn_kwargs
