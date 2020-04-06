# post_process.py
import os
import datetime
import config
# from hanspell import spell_checker

class PostProcess():
	def __init__(self, input_path, output_path):
		# load file to process
		self.input_path = input_path
		self.file = open(input_path,'rt',encoding='utf8')
		self.output_path = output_path

		# rule set
		with open(config.post_process_rule_path,'rt',encoding='utf8') as f:
			self.rules = dict(map(lambda x:tuple(x.strip().split('\t')),f))


	def rep_wrong_char(self,txt):
		for k,v in self.rules.items():
			if k in txt: txt = txt.replace(k,v)
		return txt

	# def spell_correct(self,txt):
		# return spell_checker.check(txt)['checked']
		
	def tag_strange_txt(self, txt):
		conditions = [':' in txt.replace('y_pred:',''), ]
		if any(conditions):
			txt += ' [주의]'
		return txt

	def post_process(self):
		txt_list = []
		for line in self.file:
			new_line = line.strip()
			if line.startswith('y_pred'): 
				new_line = self.rep_wrong_char(new_line)
				new_line = self.tag_strange_txt(new_line) 
			txt_list.append(new_line)
		return '\n'.join(txt_list)

	def run(self):
		with open(self.output_path,'wt',encoding='utf8') as f:
			f.write(self.post_process())
		self.file.close()



def main():
	root_dir = os.getcwd()
	input_path = os.path.join(root_dir,'pretrain_result.txt')
	now = datetime.datetime.now()
	proc = PostProcess(input_path=input_path, output_path = 'processed_{:02d}{:02d}{:02d}{:02d}.txt'.format(now.month,now.day,now.hour,now.minute))
	proc.run()

if __name__ == '__main__':
	main()