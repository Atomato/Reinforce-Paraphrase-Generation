import os
import io
import random
import datetime
import pyjosa
from kobert.utils import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
import nltk

# custom modules
import config

class PostProcess():
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

		# rule set
		with open(config.post_process_rule_path,'rt',encoding='utf8') as f:
			self.rules = dict(map(lambda x:tuple(x.strip('\n').split('\t')),f))


		#dict to store (x,y,y_pred) triplet
		self.idx_map = ['x','y','y_pred']
		self.inst_dict = {}

		# numbers / hipen
		self.num_2_txt = {'^\(1\)':['우선,','먼저,','처음으로,'],
						  '^\(2\)':['두 번째로,', '이어서,','다음으로,'],
						  '^\(3\)':['세 번째로,','이어서,','다음으로,'],
						  '^\(4\)':['네 번째로,','이어서,','다음으로,'],
						  '^\(5\)':['다섯 번째로,','이어서,','다음으로,'],
						  '^\(6\)':['여섯 번째로,','이어서,','다음으로,']
                          }


	def map_dict_by_idx(self, idx, txt):
		idx = idx % 4
		if idx == 3: return
		self.inst_dict[self.idx_map[idx]] = txt.replace(self.idx_map[idx]+':','').strip()

	def process_by_idx(self, idx, txt, orig_txt):
		if idx % 4 == 2: # y_pred lines
			txt = self.correct_numbering(txt.replace('y_pred:\t',''), orig_txt.replace('x:\t',''))
			txt = self.replace_colon(txt)
			txt = self.rep_wrong_char(txt)
			txt = self.tag_strange_txt(txt)
		return txt
	
	def rep_wrong_char(self,txt):
		for k,v in self.rules.items():
			if k in txt: txt = txt.replace(k,v)
		return txt
		
	def tag_strange_txt(self, txt):
		conditions = [':' in txt.replace('y_pred:',''), # contains : inside decoded texts 
					len(self.tokenizer(self.inst_dict['x']))> config.max_dec_steps, # input sequence length exceeds max length in decoder step
					]
		if any(conditions):
			txt += ' [주의]'
		return txt

	def correct_numbering(self, txt, orig_txt):
		for num_key in self.num_2_txt:
			orig_match = re.compile(num_key).match(orig_txt)
			print(orig_txt, txt)
			if orig_match:
				after_korean_match = re.compile('^[\ 가-힣]+,').match(txt)
				after_num_match = re.compile('^\([0-9]\)+').match(txt)
				if after_korean_match and after_korean_match.group() not in self.num_2_txt[num_key]: 
					txt = random.choice(self.num_2_txt[num_key])+txt[after_korean_match.span()[1]:]
					return txt
				elif not after_korean_match and after_num_match:
					txt = random.choice(self.num_2_txt[num_key])+txt[after_num_match.span()[1]:]
					return txt
                
		dash_match = re.compile('^-').match(txt)
		if dash_match:
			txt = txt[dash_match.span()[1]:]
            
		return txt

	def replace_colon(self, txt):
		txt = txt.replace('y_pred:\t','')
		spl = list(map(lambda x : x.strip(), txt.split(':')))
		if len(spl) != 2 : return txt # if it has more than 2 colons, just return it as-is
		if spl[0] in spl[1]: txt = '' # if second part contains the key word, remove the key word
		else: txt = '{}(은)는 '.format(spl[0])
		txt += '{}'.format(spl[1])
		txt = pyjosa.replace_josa(txt)
		return txt

	def post_process(self):
		inst_list = []
		bleu_list = []
		orig_txt = ''
		for idx,line in enumerate(self.file):
			line = line.strip()
			if idx % 4 == 0: orig_txt = line
			line = self.process_by_idx(idx,line,orig_txt)
			self.map_dict_by_idx(idx,line)
			if idx % 4 == 3: # if reading an instance ends
				sent_bleu = nltk.translate.bleu_score.sentence_bleu([self.tokenizer(self.inst_dict['y'])],
														self.tokenizer(self.inst_dict['y_pred']), 
														weights=(0.5,0.5))
				bleu_list.append(sent_bleu)
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

	def return_processed_file(self):
		inst_list = self.post_process()
		ret = {'x':[],'y':[],'y_pred':[]}
		for _dict in inst_list[:-1]:
			for k,v in _dict.items(): ret[k].append('{}'.format(v))
		if self.is_filetype(self.file): self.file.close()
		return ret



def main():
	root_dir = os.getcwd()
	input_path = os.path.join(root_dir,'new_hyperparam_result.txt')
	now = datetime.datetime.now()
	proc = PostProcess(input_path_or_input_list=input_path, output_path = 'processed_{:02d}{:02d}{:02d}{:02d}.txt'.format(now.month,now.day,now.hour,now.minute))
	proc.write_processed_file()

if __name__ == '__main__':
	main()