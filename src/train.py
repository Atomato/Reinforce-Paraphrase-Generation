from __future__ import unicode_literals, print_function, division

import os
import time
import argparse
import shutil

import torch
import numpy as np
from model import Model
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Adagrad

import tensorflow as tf
# if tensorflow version is over v2, disable v2 behavior
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import config
from batcher import Batcher
from data import Vocab
from train_util import *
from utils import calc_running_avg_loss, print_log
from eval import Evaluate

use_cuda = config.use_gpu and torch.cuda.is_available()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(5)
        
        if not os.path.exists(config.log_root):
            os.makedirs(config.log_root)

        self.model_dir = os.path.join(config.log_root, 'train_model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        
        self.eval_log = os.path.join(config.log_root, 'eval_log')
        if not os.path.exists(self.eval_log):
            os.mkdir(self.eval_log)
        self.summary_writer = tf.compat.v1.summary.FileWriter(self.eval_log)


    def save_model(self, running_avg_loss, iter, mode):
        state = {
            'iter': iter,
            'kogpt2_state_dict': self.model.kogpt2.state_dict(),
            'current_loss': running_avg_loss
        }
        if mode == 'train':
            save_model_dir = self.model_dir
        else:
            best_model_dir = os.path.join(config.log_root, 'best_model')
            if not os.path.exists(best_model_dir):
                os.mkdir(best_model_dir)
            save_model_dir = best_model_dir
        
        if len(os.listdir(save_model_dir))>0:
            shutil.rmtree(save_model_dir)
            time.sleep(2)
            os.mkdir(save_model_dir)
        train_model_path = os.path.join(save_model_dir, 'model_best_%d'%(iter))
        torch.save(state, train_model_path)
        return train_model_path
    
    
    def print_trainable_params(self, model):
        for name, params in model.named_parameters():
            # print(name, params.requires_grad)
            if params.requires_grad:
                print(name)
    

    def setup_train(self, model_file_path=None, vocab = None, log=None):
        self.model = Model(model_file_path)
        params = list(self.model.kogpt2.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        if config.mode == 'MLE':
            self.optimizer = Adagrad(params, lr=0.15, initial_accumulator_value=0.1)
        else:
            self.optimizer = Adam(params, lr=initial_lr)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']
        return start_iter, start_loss
    

    def train_one_batch(self, batch, alpha, beta):
        enc_dec_batch, enc_dec_padding_mask, max_enc_dec_len, enc_dec_lens_var, enc_dec_target_batch = \
            get_inout_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()
        
        final_dist_batch, _ = self.model.kogpt2(enc_dec_batch) # B x L x V
        probs = torch.gather(final_dist_batch, 2, enc_dec_target_batch.unsqueeze(2)).squeeze(2) # B x L
        step_nll = -torch.log(probs + config.eps) # B x L
        batch_loss = torch.sum(step_nll, dim=1)  # B
        loss = torch.mean(batch_loss)

        avg_reward = torch.zeros(1)
        loss.backward()

        self.norm = clip_grad_norm_(self.model.kogpt2.parameters(), config.max_grad_norm)

        self.optimizer.step()
        return loss.item(), avg_reward.item()


    def trainIters(self, n_iters, model_file_path=None):
        if config.mode not in ["MLE", "RL", "GTI", "SO", "SIO", "DAGGER", "DAGGER*"]:
            print("\nTRAINING MODE ERROR\n")
            raise ValueError
        # log file path
        log_path = os.path.join(config.log_root, 'log')
        log = open(log_path, 'w')
        print_log("==============================", file=log)
        iter, running_avg_loss = self.setup_train(model_file_path, vocab=self.vocab, log=log)
        min_val_loss = np.inf
        
        alpha = config.alpha
        beta = config.beta
        k1 = config.k1
        k2 = config.k2
        delay = iter # set to 0 in the original code (wyu-du)

        print("\nLog root is %s" % config.log_root)
        print_log("Train mode is %s" % config.mode, file=log)
        print_log("k1: %s, k2: %s" % (config.k1, config.k2), file=log)
        print_log("==============================", file=log)

        cur_time = time.time()
        while iter < n_iters:
            if config.mode == 'RL':
                alpha = 0.
                beta = 0.
            elif config.mode == 'GTI':
                alpha = 1.
                beta = 0.
            elif config.mode == 'SO':
                alpha = 1.
                beta = k2/(k2+np.exp((iter-delay)/k2))
            elif config.mode == 'SIO':
                alpha *= k1
                if alpha < 0.01:
                    beta = k2/(k2+np.exp((iter-delay)/k2))
                else:
                    beta = 1.
                    delay += 1
            elif config.mode == 'DAGGER':
                alpha *= k1
                beta = 1.
            elif config.mode == 'DAGGER*':
                alpha = config.alpha
                beta = 1.
            else:
                alpha = 1.
                beta = 1.
            
            batch = self.batcher.next_batch()
            loss, avg_reward = self.train_one_batch(batch, alpha, beta)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1
            
            if iter % config.print_interval == 0:
                print_log('steps %d, current_loss: %f, avg_reward: %f, alpha: %f, beta: %f, delay: %d' % \
                            (iter, loss, avg_reward, alpha, beta, delay), file=log)
            
            if iter % config.save_model_iter == 0:
                model_file_path = self.save_model(running_avg_loss, iter, mode='train')
                evl_model = Evaluate(model_file_path)
                val_avg_loss = evl_model.run_eval()
                if val_avg_loss < min_val_loss:
                    min_val_loss = val_avg_loss
                    best_model_file_path = self.save_model(running_avg_loss, iter, mode='eval')
                    print_log('Save best model at %s' % best_model_file_path, file=log)
                print_log('steps %d, train_loss: %f, val_loss: %f, time: %ds' % \
                                        (iter, loss, val_avg_loss, time.time()-cur_time), file=log)
                # write val_loss into tensorboard
                loss_sum = tf.compat.v1.Summary()
                loss_sum.value.add(tag='val_avg_loss', simple_value=val_avg_loss)
                self.summary_writer.add_summary(loss_sum, global_step=iter)
                self.summary_writer.flush()
                cur_time = time.time()

                # To evade Out of Memory error
                del evl_model

        log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()
    
    train_processor = Train()
    train_processor.trainIters(config.max_iterations, args.model_file_path)
