from __future__ import unicode_literals, print_function, division

import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from kogpt2_mem.pytorch_kogpt2 import get_pytorch_kogpt2_model

use_cuda = config.use_gpu and torch.cuda.is_available()
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

class KoGPT2(nn.Module):
    def __init__(self):
        super(KoGPT2, self).__init__()
        self.kogpt2, _ = get_pytorch_kogpt2_model() # kogpt2 model, vocab

    def forward(self, input_ids, past=None):
        pred, past = self.kogpt2(input_ids=input_ids, past=past) # B x L x V
        vocab_dist = F.softmax(pred, dim=2) # B x L x V
        return vocab_dist, past

class Model(object):
    def __init__(self, model_file_path=None, is_eval=False):
        kogpt2 = KoGPT2()
        
        if is_eval:
            self.kogpt2 = kogpt2.eval()
        else:
            self.kogpt2 = kogpt2.train()

        if use_cuda:
            self.kogpt2 = kogpt2.cuda()

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.kogpt2.load_state_dict(state['kogpt2_state_dict'])
