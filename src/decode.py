# Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/
from __future__ import unicode_literals, print_function, division

import sys
import os
import time

import torch
from torch.autograd import Variable
import nltk
import numpy as np

import data, config
from batcher import Batcher
from data import Vocab
from model import Model
from utils import write_for_rouge, rouge_eval, rouge_log, write_for_result
from train_util import get_input_from_batch

from post_process import PostProcess

use_cuda = config.use_gpu and torch.cuda.is_available()


class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path, data_path, data_class='val'):
        self.data_class = data_class
        if self.data_class not in  ['val', 'test']:
            print("data_class must be 'val' or 'test'.")
            raise ValueError

        # model_file_path e.g. --> ../log/{MODE NAME}/best_model/model_best_XXXXX
        model_name = os.path.basename(model_file_path)
        # log_root e.g. --> ../log/{MODE NAME}/
        log_root = os.path.dirname(os.path.dirname(model_file_path))
        # _decode_dir e.g. --> ../log/{MODE NAME}/decode_model_best_XXXXX/
        self._decode_dir = os.path.join(log_root, 'decode_%s' % (model_name))
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        self._result_path = os.path.join(self._decode_dir, 'result_%s_%s.txt' \
                                                        % (model_name, self.data_class))
        # remove result file if exist
        if os.path.isfile(self._result_path):
            os.remove(self._result_path)
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        time.sleep(5)

        self.model = Model(model_file_path, is_eval=True)


    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def beam_search(self, batch):
        # batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_input_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0 # 1 x 2H
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]

            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []
            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)

            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)
        return beams_sorted[0]
    
    
    def decode(self):
        start = time.time()
        counter = 0
        bleu_scores = []
        batch = self.batcher.next_batch()
        while batch is not None:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_articles = batch.original_articles[0]
            original_abstracts = batch.original_abstracts_sents[0]
            reference = original_abstracts[0].strip().split()
            bleu = nltk.translate.bleu_score.sentence_bleu([reference], decoded_words, weights = (0.5, 0.5))
            bleu_scores.append(bleu)

            # write_for_rouge(original_abstracts, decoded_words, counter,
            #                 self._rouge_ref_dir, self._rouge_dec_dir)

            write_for_result(original_articles, original_abstracts, decoded_words, \
                                                self._result_path, self.data_class)

            counter += 1
            if counter % 1000 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()

            batch = self.batcher.next_batch()
        
        '''
        # uncomment this if you successfully install `pyrouge`
        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)
        '''

        if self.data_class == 'val':
            print('Average BLEU score:', np.mean(bleu_scores))
            with open(self._result_path, "a") as f:
                print('Average BLEU score:', np.mean(bleu_scores), file=f)

    def get_processed_path(self):
        # ../log/{MODE NAME}/decode_model_best_XXXXX/result_model_best_2800_{data_class}.txt
        input_path = self._result_path
        temp = os.path.splitext(input_path)
        # ../log/{MODE NAME}/decode_model_best_XXXXX/result_model_best_2800_{data_class}_processed.txt
        output_path = temp[0] + "_processed" + temp[1]
        return input_path, output_path

if __name__ == '__main__':
    model_filename = sys.argv[1]
    beam_Search_processor_val = BeamSearch(model_filename, config.eval_data_path)
    print('Decoding validation set...')
    beam_Search_processor_val.decode()
    print('Done!\n')


    beam_Search_processor_test = BeamSearch(model_filename, config.decode_data_path,
                                                                data_class='test')
    print('Decoding test set...')
    beam_Search_processor_test.decode()
    print('Done!\n')

    print('Post-processing...')
    input_val, output_val = beam_Search_processor_val.get_processed_path()
    proc_val = PostProcess(input_val, output_val)
    proc_val.write_processed_file()

    input_test, output_test = beam_Search_processor_test.get_processed_path()
    proc_test = PostProcess(input_test, output_test)
    proc_test.write_processed_file()
    print('Done!\n')
