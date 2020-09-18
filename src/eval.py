from __future__ import unicode_literals, print_function, division

import time
import sys
import numpy as np
import torch

import config
from batcher import Batcher
from data import Vocab
from train_util import get_input_from_batch, get_output_from_batch, get_inout_from_batch
from model import Model


use_cuda = config.use_gpu and torch.cuda.is_available()


class Evaluate(object):
    def __init__(self, model_file_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.eval_data_path, self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=True)
        self.model_file_path = model_file_path
        time.sleep(5)

        self.model = Model(model_file_path, is_eval=True)


    def eval_one_batch(self, batch):
        enc_dec_batch, enc_dec_padding_mask, max_enc_dec_len, enc_dec_lens_var, enc_dec_target_batch = \
            get_inout_from_batch(batch, use_cuda)
            
        with torch.no_grad():
            final_dist_batch, _ = self.model.kogpt2(enc_dec_batch) # B x L x V
            gold_probs = torch.gather(final_dist_batch, 2, enc_dec_target_batch.unsqueeze(2)).squeeze() # B x L
            step_loss = -torch.log(gold_probs + config.eps) # B x L
            step_mask = enc_dec_padding_mask # B x L
            step_loss = step_loss * step_mask

        sum_step_losses = torch.sum(step_loss, 1)
        batch_avg_loss = sum_step_losses / enc_dec_lens_var

        loss = torch.mean(batch_avg_loss)
        return loss.item()


    def run_eval(self):
        batch = self.batcher.next_batch()
        loss_list = []
        while batch is not None:
            loss = self.eval_one_batch(batch)
            loss_list.append(loss)
            batch = self.batcher.next_batch()
        return np.mean(loss_list)
    

if __name__ == '__main__':
    # model_filename = "../log/KoGPT2/best_model/model_best_400"
    model_filename = sys.argv[1]
    eval_processor = Evaluate(model_filename)
    eval_processor.run_eval()
