from __future__ import unicode_literals, print_function, division

import sys
import os
import time

import torch
from torch.autograd import Variable
import nltk

import data, config
from data import Vocab
from model import Model
from utils import write_for_rouge, rouge_eval, rouge_log, write_for_result
from train_util import get_input_from_batch

import queue as Queue
import time
from random import shuffle
from threading import Thread
import numpy as np

use_cuda = config.use_gpu and torch.cuda.is_available()


import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
import glob
import re
from kobert.utils import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

SPECIAL_TOKENS = ['<EXPR>', '<UNVAR>', '<ARRW>', '<EQUL>', '<INEQ>']
KOREAN_2_SPECIAL = {'(수식)':'\N{Arabic Poetic Verse Sign}',
                     '(미지수)':'\N{Arabic Sign Misra}' ,
                     '(화살표)':'\N{Arabic Place of Sajdah}',
                     '(등호)':'\N{Arabic Sign Sindhi Ampersand}',
                     '(부등호)':'\N{ARABIC SEMICOLON}'}
SPECIAL_2_ENG = dict(zip(['\N{Arabic Poetic Verse Sign}',
                     '\N{Arabic Sign Misra}',
                     '\N{Arabic Place of Sajdah}',
                     '\N{Arabic Sign Sindhi Ampersand}',
                     '\N{ARABIC SEMICOLON}'], SPECIAL_TOKENS))

HIGHLIGHT = "▁ @ h i g h l i g h t"

TOK_PATH = get_tokenizer()
sp = SentencepieceTokenizer(TOK_PATH)

"""import random
random.seed(1234)"""
def kobert_tokenizer(sentence):
    for k,v in KOREAN_2_SPECIAL.items(): # replace special tokens
        sentence = sentence.replace(k,v)
    tokens = [token for token in sp(sentence)]
    tokens = [SPECIAL_2_ENG[ele] if ele in SPECIAL_2_ENG else ele for ele in tokens]
    # tokens = [token for token in tokens if token != "▁"]

    return ' '.join(tokens)


# Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/
class Example(object):
    def __init__(self, article, abstract_sentences, vocab):
        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)

        # Process the article
        article_words = article.split()
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        self.enc_len = len(article_words)  # store the length after truncation but before padding
        self.enc_input = [vocab.word2id(w) for w in
                          article_words]  # list of word ids; OOVs are represented by the id for UNK token

        # Process the abstract
        abstract = ' '.join(abstract_sentences)
        abstract_words = abstract.split()  # list of strings
        abs_ids = [vocab.word2id(w) for w in
                   abstract_words]  # list of word ids; OOVs are represented by the id for UNK token

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding,
                                                                 stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if config.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            # NOTE: dec_input does not contain article OOV ids!!!!
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding,
                                                        stop_decoding)

        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
  def __init__(self, example_list, vocab, batch_size):
    self.batch_size = batch_size
    self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.init_encoder_seq(example_list) # initialize the input to the encoder
    self.init_decoder_seq(example_list) # initialize the input and targets for the decoder
    self.store_orig_strings(example_list) # store the original strings


  def init_encoder_seq(self, example_list):
    # Determine the maximum length of the encoder input sequence in this batch
    max_enc_seq_len = max([ex.enc_len for ex in example_list])

    # Pad the encoder input sequences up to the length of the longest sequence
    for ex in example_list:
      ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

    # Initialize the numpy arrays
    # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
    self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
    self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
    self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.enc_batch[i, :] = ex.enc_input[:]
      self.enc_lens[i] = ex.enc_len
      for j in range(ex.enc_len):
        self.enc_padding_mask[i][j] = 1

    # For pointer-generator mode, need to store some extra info
    if config.pointer_gen:
      # Determine the max number of in-article OOVs in this batch
      self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
      # Store the in-article OOVs themselves
      self.art_oovs = [ex.article_oovs for ex in example_list]
      # Store the version of the enc_batch that uses the article OOV ids
      self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
      for i, ex in enumerate(example_list):
        self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]


  def init_decoder_seq(self, example_list):
    # Pad the inputs and targets
    for ex in example_list:
      ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

    # Initialize the numpy arrays.
    self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
    self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
    self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
    self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.dec_batch[i, :] = ex.dec_input[:]
      self.target_batch[i, :] = ex.target[:]
      self.dec_lens[i] = ex.dec_len
      for j in range(ex.dec_len):
        self.dec_padding_mask[i][j] = 1


  def store_orig_strings(self, example_list):
    self.original_articles = [ex.original_article for ex in example_list] # list of lists
    self.original_abstracts = [ex.original_abstract for ex in example_list] # list of lists
    self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list]



class Batcher(object):
  BATCH_QUEUE_MAX = 10 # max number of batches the batch_queue can hold
  def __init__(self, sentence, vocab, mode, batch_size, single_pass):
    self._sentence = sentence
    self._vocab = vocab
    self._single_pass = single_pass
    self.mode = mode
    self.batch_size = batch_size
    # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
    self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
    self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)

    # Different settings depending on whether we're in single_pass mode or not
    if single_pass:
      self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
      self._num_batch_q_threads = 1  # just one thread to batch examples
      self._bucketing_cache_size = 1 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
      self._finished_reading = False # this will tell us when we're finished reading the dataset
    else:
      self._num_example_q_threads = 1 #16 # num threads to fill example queue
      self._num_batch_q_threads = 1 #4  # num threads to fill batch queue
      self._bucketing_cache_size = 1 #100 # how many batches-worth of examples to load into cache before bucketing

    # Start the threads that load the queues
    self._example_q_threads = []
    for _ in range(self._num_example_q_threads):
      self._example_q_threads.append(Thread(target=self.fill_example_queue))
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()
    self._batch_q_threads = []
    for _ in range(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()

    # Start a thread that watches the other threads and restarts them if they're dead
    if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
      self._watch_thread = Thread(target=self.watch_threads)
      self._watch_thread.daemon = True
      self._watch_thread.start()


  def next_batch(self):
    # If the batch queue is empty, print a warning
    if self._batch_queue.qsize() == 0:
      #tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
      if self._single_pass and self._finished_reading:
        #tf.logging.info("Finished reading dataset in single_pass mode.")
        return None

    batch = self._batch_queue.get() # get the next Batch
    return batch


  def fill_example_queue(self):
    print("fill_example")
    input_gen = self.text_generator(self._sentence)
    while True:
      try:
        (article, abstract) = next(input_gen) # read the next example from file. article and abstract are both strings.
      except StopIteration: # if there are no more examples:
        if self._single_pass:
          self._finished_reading = True
          break
        else:
          raise Exception("single_pass mode is off but the example generator is out of data; error.")
          break
#      abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)] # Use the <s> and </s> tags in abstract to get a list of sentences.
#      abstract = str(abstract, encoding='utf8')
      abstract_sentences = [abstract]
      example = Example(article, abstract_sentences, self._vocab) # Process into an Example.
      self._example_queue.put(example) # place the Example in the example queue.


  def fill_batch_queue(self):
    while True:
      if self.mode == 'decode':
        # beam search decode mode single example repeated in the batch
        ex = self._example_queue.get()
        b = [ex for _ in range(self.batch_size)]
        self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
      else:
        # Get bucketing_cache_size-many batches of Examples into a list, then sort
        inputs = []
        for _ in range(self.batch_size * self._bucketing_cache_size):
          inputs.append(self._example_queue.get())
        inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True) # sort by length of encoder sequence

        # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
        batches = []
        for i in range(0, len(inputs), self.batch_size):
          batches.append(inputs[i:i + self.batch_size])
        if not self._single_pass:
          shuffle(batches)
        for b in batches:  # each b is a list of Example objects
          self._batch_queue.put(Batch(b, self._vocab, self.batch_size))


  def watch_threads(self):
    while True:
#      tf.logging.info(
#        'Bucket queue size: %i, Input queue size: %i',
#        self._batch_queue.qsize(), self._example_queue.qsize())

      time.sleep(60)
      for idx,t in enumerate(self._example_q_threads):
        if not t.is_alive(): # if the thread is dead
          #tf.logging.error('Found example queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_example_queue)
          self._example_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self._batch_q_threads):
        if not t.is_alive(): # if the thread is dead
          #tf.logging.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_batch_queue)
          self._batch_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()


  def text_generator(self, sentence):
     yield (sentence, sentence)

class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    context=context,
                    coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path, sentence, data_class='test'):
        self.data_class = data_class
        if self.data_class not in ['val', 'test']:
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
        self.batcher = Batcher(sentence, self.vocab, mode='decode',
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

        dec_h, dec_c = s_t_0  # 1 x 2H
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_0[0],
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
            all_state_h = []
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
                                                                                    encoder_outputs, encoder_feature,
                                                                                    enc_padding_mask, c_t_1,
                                                                                    extra_zeros, enc_batch_extend_vocab,
                                                                                    coverage_t_1, steps)

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
        batch = self.batcher.next_batch()

        while batch is not None:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))
            print("".join(decoded_words).replace("▁", " ").strip())
            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_articles = batch.original_articles[0]

            original_abstracts = batch.original_abstracts_sents[0]
            #print(original_abstracts)
            reference = original_abstracts[0].strip().split()

            # write_for_rouge(original_abstracts, decoded_words, counter,
            #                 self._rouge_ref_dir, self._rouge_dec_dir)

            counter += 1
            if counter % 1000 == 0:
                print('%d example in %d sec' % (counter, time.time() - start))
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


"""if __name__ == '__main__':
    sentence = kobert_tokenizer("(수식)를 계산하여 보자.")
    model_filename = "../log/MLE/best_model/model_best_2000"
    beam_Search_processor = BeamSearch(model_filename, sentence)
    beam_Search_processor.decode()
    print('Done!\n')"""
if __name__ == '__main__':
    #model_filename = sys.argv[1]
    model_filename = '../log/MLE/best_model/model_best_2000'

    """ beam_Search_processor_val = BeamSearch(model_filename, config.eval_data_path)
    print('Decoding validation set...')
    beam_Search_processor_val.decode()
    print('Done!\n')"""

    sentence = kobert_tokenizer("(수식)를 계산하여 보자.")
    beam_Search_processor_test = BeamSearch(model_filename, sentence, data_class='test')
    print('Decoding test set...')
    beam_Search_processor_test.decode()
    print('Done!\n')