import os
import io
import re
import random
import datetime
import pyjosa
from kogpt2.utils import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
import nltk

# custom modules
import config

from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from model import Model
import torch

TO_TOKEN = {'(수식)': '<expr>', '(미지수)': '<unvar>', '(화살표)': '<arrw>', '(등호)': '<equl>'}
FROM_TOKEN = {'<expr>': '(수식)', '<unvar>': '(미지수)', '<arrw>': '(화살표)', '<equl>': '(등호)'}

class Greedy():
    def __init__(self, input_path_or_input_list, output_path):
        # load file to process
        if isinstance(input_path_or_input_list, str): # if a path is given as path string
            self.file = open(input_path_or_input_list,'rt',encoding='utf8')
        else: # if a path is given as list
            self.file = input_path_or_input_list
        self.output_path = output_path
        self.is_filetype = lambda x: any([isinstance(x, io.TextIOBase),
                                            isinstance(x, io.BufferedIOBase),
                                            isinstance(x, io.RawIOBase),
                                            isinstance(x, io.IOBase)])

        # tokenizer
        tok_path = get_tokenizer()
        self.tokenizer = SentencepieceTokenizer(tok_path)

        #dict to store (x,y,y_pred) triplet
        self.idx_map = ['x','y','y_pred']
        self.inst_dict = {}

        _, self.vocab = get_pytorch_kogpt2_model()
        model_file_path = "../log/KoGPT2_fine/best_model/model_best_3300"
        self.model = Model(model_file_path, is_eval=True)
        self.model = self.model.kogpt2.cpu()

    def inference(self, txt):
        toked = self.tokenizer(txt)
        toked = [self.vocab.bos_token] + toked + [self.vocab.eos_token]
        sent = ''
        for _ in range(150):
          input_ids = torch.tensor(self.vocab[toked]).unsqueeze(0)
          pred = self.model.kogpt2(input_ids=input_ids)[0]

          gen = self.vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]
          if gen == '</s>':
              break
          sent += gen.replace('▁', ' ')
          toked += [gen]
          
        return sent

    def map_dict_by_idx(self, idx, txt):
        idx = idx % 4
        if idx == 3: return
        self.inst_dict[self.idx_map[idx]] = txt.replace(self.idx_map[idx]+':','').strip()

    def process_by_idx(self, idx, txt, orig_txt):
        if idx % 4 == 2: # y_pred lines
            txt = self.inference(orig_txt.replace('x:\t',''))
        return txt

    def to_token(self, txt):
        for k, v in TO_TOKEN.items():
            txt = txt.replace(k, v)
        return txt

    def from_token(self, txt):
        for k, v in FROM_TOKEN.items():
            txt = txt.replace(k, v)
        return txt

    def post_process(self):
        inst_list = []
        bleu_list = []
        orig_txt = ''
        for idx,line in enumerate(self.file):
            line = line.strip()
            line = self.from_token(line)
            if idx % 4 == 0: orig_txt = line
            line = self.process_by_idx(idx,line,orig_txt)
            # line = self.to_token(line)
            self.map_dict_by_idx(idx,line)
            if idx % 4 == 3: # if reading an instance ends
                sent_bleu = nltk.translate.bleu_score.sentence_bleu([self.tokenizer(self.inst_dict['y'])],
                                                        self.tokenizer(self.inst_dict['y_pred']), 
                                                        weights=(0.5,0.5))
                bleu_list.append(sent_bleu)
                # print(self.inst_dict)
                inst_list.append(self.inst_dict)
                self.inst_dict = {} # flush instance dict
        # record the bleu score of post-processed predictions
        inst_list.append('Average BLEU score: {:f}'.format(sum(bleu_list)/len(bleu_list)))
        return inst_list

    def write_processed_file(self):
        # write texts
        inst_list = self.post_process()
        txt = ''
        for _dict in inst_list[:-1]:
            for k,v in _dict.items(): txt += '{}:\t{}\n'.format(k,v)
            txt += '\n'
        txt += inst_list[-1]

        with open(self.output_path,'wt',encoding='utf8') as f:
            f.write(txt)
        if self.is_filetype(self.file): self.file.close()

def main():
    root_dir = os.getcwd()
    input_path = os.path.join(root_dir,'result_model_best_5500_val.txt')
    now = datetime.datetime.now()
    greedy = Greedy(input_path_or_input_list=input_path, output_path = 'processed_{:02d}{:02d}{:02d}{:02d}.txt'.format(now.month,now.day,now.hour,now.minute))
    greedy.write_processed_file()

if __name__ == '__main__':
    main()