import os
import random
import datetime
from kobert.utils import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer

# custom modules
import config

class PostProcess():
	def __init__(self, input_path, output_path):
		# load file to process
		self.input_path = input_path
		self.file = open(input_path,'rt',encoding='utf8')
		self.output_path = output_path

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
		self.num_2_txt = {'(1)': ['우선,','먼저,','처음으로,'],
						  '(2)':['이어서,','다음으로,',''],
						  '(3)':['이어서,','다음으로,',''],
						  '(4)':['이어서,','다음으로,',''],
						  '(5)':['이어서,','다음으로,',''],
						  '-' : ['']}


	def map_dict_by_idx(self, idx, txt):
		idx = idx % 4
		if idx == 3: return
		self.inst_dict[self.idx_map[idx]] = txt.replace(self.idx_map[idx]+':','').strip()


	def process_by_idx(self, idx, txt):
		if idx % 4 == 2: # y_pred lines
			txt = self.correct_numbering(txt)
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

	def correct_numbering(self, txt):
		for k,v in self.num_2_txt.items():
			if k in txt: txt = txt.replace(k,random.choice(v))
		return txt

	def post_process(self):
		inst_list = []
		for idx,line in enumerate(self.file):
			line = line.strip()
			line = self.process_by_idx(idx,line)
			self.map_dict_by_idx(idx,line)
			if idx % 4 == 3: # if reading an instance ends
				inst_list.append(self.inst_dict)
				self.inst_dict = {} # flush instance dict

		inst_list.append(self.inst_dict)
		return inst_list

	def run(self):
		# write texts
		inst_list = self.post_process()
		txt = ''
		for _dict in inst_list[:-1]:
			for k,v in _dict.items(): txt += '{}: {}\n'.format(k,v)
			txt += '\n'
		txt += inst_list[-1]['x']

		with open(self.output_path,'wt',encoding='utf8') as f:
			f.write(txt)
		self.file.close()



def main():
	root_dir = os.getcwd()
	input_path = os.path.join(root_dir,'pretrain_result.txt')
	now = datetime.datetime.now()
	proc = PostProcess(input_path=input_path, output_path = 'processed_{:02d}{:02d}{:02d}{:02d}.txt'.format(now.month,now.day,now.hour,now.minute))
	proc.run()

if __name__ == '__main__':
	main()