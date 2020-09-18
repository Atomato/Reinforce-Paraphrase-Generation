# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:34:25 2020

@author: LSH
"""

from __future__ import unicode_literals, print_function, division
import torch
# from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from KoGPT2_ELMo import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer

import torch.nn.functional as F

import config
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


tok_path = get_tokenizer()
model_path = "../model_best_3300"
kogpt2, vocab = get_pytorch_kogpt2_model()

state = torch.load(model_path, map_location= lambda storage, location: storage)
state_dict = {k[7:]:v for k,v in state['kogpt2_state_dict'].items()}
kogpt2.load_state_dict(state_dict)

for name, param in kogpt2.named_parameters():
    print(name, param.shape)

for name, param in kogpt2.lm_head.named_parameters():
    print(name, param.shape, param.requires_grad)

params = list(kogpt2.named_parameters())

tok = SentencepieceTokenizer(tok_path)
sent = '2019년 한해를 보내며,'
toked = tok(sent)

input_ids = torch.tensor([vocab[vocab.bos_token],]  + vocab[toked]).unsqueeze(0)
pred = kogpt2(input_ids)

kogpt2_layers = kogpt2.config.n_layer+1
lambda_i = nn.Parameter(torch.Tensor(kogpt2_layers).fill_(1.0), requires_grad=True)
gamma = nn.Parameter(torch.Tensor(1).fill_(1.0), requires_grad=True)
z = torch.sum(torch.exp(lambda_i), 0)

layer_norm = nn.LayerNorm(pred[2][0].shape[1:], elementwise_affine=False)

out_sum = torch.Tensor()
for i, out in enumerate(pred[2]):
    normalized_out = layer_norm(out)
    out_sum = torch.cat((out_sum, normalized_out.unsqueeze(0).transpose(0,1)), 1)
    
ELMo = torch.sum(torch.mul((lambda_i/z).unsqueeze(1), out_sum.transpose(1,2)), 2) * gamma
lm_logits = kogpt2.lm_head(ELMo)

vocab_dist = F.softmax(lm_logits, dim=2) # B x L x V(50000)
