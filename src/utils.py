# Content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/
import os
# uncomment this if you successfully install `pyrouge`
# import pyrouge
import logging

SPECIAL_TOKENS = ['<expr>', '<unvar>', '<equl>', '<arrw>']

def print_results(article, abstract, decoded_output):
  print ("")
  print('ARTICLE:  %s', article)
  print('REFERENCE SUMMARY: %s', abstract)
  print('GENERATED SUMMARY: %s', decoded_output)
  print( "")


def make_html_safe(s):
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s


def rouge_eval(ref_dir, dec_dir):
  r = pyrouge.Rouge155()
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  rouge_results = r.convert_and_evaluate()
  return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
  log_str = ""
  for x in ["1","2","l"]:
    log_str += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_%s_%s" % (x,y)
      key_cb = key + "_cb"
      key_ce = key + "_ce"
      val = results_dict[key]
      val_cb = results_dict[key_cb]
      val_ce = results_dict[key_ce]
      log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
  print(log_str)
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
  print("Writing final ROUGE results to %s..."%(results_file))
  with open(results_file, "w") as f:
    f.write(log_str)


def calc_running_avg_loss(loss, running_avg_loss, step, decay=0.99):
  if running_avg_loss == 0:  # on the first iteration just take the loss
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)  # clip
  return running_avg_loss


def write_for_rouge(reference_sents, decoded_words, ex_index,
                    _rouge_ref_dir, _rouge_dec_dir):
  decoded_sents = []
  while len(decoded_words) > 0:
    try:
      fst_period_idx = decoded_words.index(".")
    except ValueError:
      fst_period_idx = len(decoded_words)
    sent = decoded_words[:fst_period_idx + 1]
    decoded_words = decoded_words[fst_period_idx + 1:]
    decoded_sents.append(' '.join(sent))

  # pyrouge calls a perl script that puts the data into HTML files.
  # Therefore we need to make our output HTML safe.
  decoded_sents = [make_html_safe(w) for w in decoded_sents]
  reference_sents = [make_html_safe(w) for w in reference_sents]

  ref_file = os.path.join(_rouge_ref_dir, "%06d_reference.txt" % ex_index)
  decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % ex_index)

  with open(ref_file, "w") as f:
    for idx, sent in enumerate(reference_sents):
      f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
  with open(decoded_file, "w") as f:
    for idx, sent in enumerate(decoded_sents):
      f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

def reverse_tokenizer(sentence):
  sents = sentence.split()

  return "".join(sents).replace("▁", " ").strip()

def write_for_result(input_sents, reference_sents, decoded_words, _result_path, data_class):
  decoded_sents = []
  while len(decoded_words) > 0:
    try:
      fst_period_idx = decoded_words.index(".")
    except ValueError:
      fst_period_idx = len(decoded_words)
    sent = decoded_words[:fst_period_idx + 1]
    decoded_words = decoded_words[fst_period_idx + 1:]
    decoded_sents.append(' '.join(sent))

  # remove "▁"
  input_s = reverse_tokenizer(input_sents)
  reference_s = reverse_tokenizer(reference_sents[0])
  decoded_s = reverse_tokenizer(decoded_sents[0])

  if os.path.isfile(_result_path):
    with open(_result_path, "a") as f:
      print("x:\t\t" + input_s, file=f)
      if data_class == 'val': print("y:\t\t" + reference_s, file=f)
      print("y_pred:\t" + decoded_s + "\n", file=f)
  else:
    with open(_result_path, "w") as f:
      print("x:\t\t" + input_s, file=f)
      if data_class == 'val': print("y:\t\t" + reference_s, file=f)
      print("y_pred:\t" + decoded_s + "\n", file=f)

def gen_ngram(sent, n=2):
    words = sent.split()
    ngrams = []
    for i, token in enumerate(words):
        if i<=len(words)-n:
            ngram = '-'.join(words[i:i+n])
            ngrams.append(ngram)
    return ngrams


def count_match(ref, dec, n=2):
    counts = 0.
    for d_word in dec:
        if d_word in ref:
            counts += 1
    return counts


def rouge_2(gold_sent, decode_sent):
    bigrams_ref = gen_ngram(gold_sent, 2)
    bigrams_dec = gen_ngram(decode_sent, 2)
    if len(bigrams_ref) == 0:
        recall = 0.
    else:
        recall = count_match(bigrams_ref, bigrams_dec, 2)/len(bigrams_ref)
    if len(bigrams_dec) == 0:
        precision = 0.
    else:
        precision = count_match(bigrams_ref, bigrams_dec, 2)/len(bigrams_dec)
    if recall+precision == 0:
        f1_score = 0.
    else:
        f1_score = 2*recall*precision/(recall+precision)
    return f1_score

# print log info on SCREEN and LOG file simultaneously
def print_log(*args, **kwargs):
    print(*args)
    if len(kwargs) > 0:
        print(*args, **kwargs)
    return None