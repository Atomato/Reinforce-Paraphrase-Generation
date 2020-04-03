# A Unified Reinforcement Learning Framework for Pointer Generator Model
This repository contains the data and code for the paper ["An Empirical Comparison on Imitation Learning and Reinforcement Learning for Paraphrase Generation"](https://arxiv.org/abs/1908.10835).

## Requirement
### Install NLTK
`pip install -U nltk`

## Useage
### Training
1. Model Setting: modify the path where the model will be saved.
```
vim config.py
log_root = os.path.join(root_dir, "Reinforce-Paraphrase-Generation/log/MLE")
```

2. Pre-train: train the standard pointer-generator model with supervised learning from scratch.
```
python train.py
```

3. Fine-tune: modify the training mode and the path where the fine-tuned model will be saved (mode: "MLE" -> "RL").
```
vim config.py
log_root = os.path.join(root_dir, "Reinforce-Paraphrase-Generation/log/RL")
mode = "RL"
```
Fine tune the pointer-generator model with REINFORCE algorithm.
```
python train.py -m ../log/MLE/best_model/model_best_XXXXX
```


### Decoding & Evaluation
1. Decoding: Apply beam search to generate sentences on test set with the model path:
```
python decode.py ../log/{MODE NAME}/best_model/model_best_XXXXX
```

2. Result: The result.txt will be made in following path:
```
../log/{MODE NAME}/decode_model_best_XXXXX/result_model_best_XXXXX.txt
```

3. Log file: The log on the train and validation loss will be made in following path:
```
../log/{MODE NAME}/log
```

4. Evaluation: 
	- The average BLEU score on the validation set will show up automatically in the terminal after finishing decoding.
	
	- If you want to get the ROUGE scores, you should first intall `pyrouge`, here is the [guidance](https://ireneli.eu/2018/01/11/working-with-rouge-1-5-5-evaluation-metric-in-python/). Then, you can uncomment the code snippet specified in `utils.py` and `decode.py`. Finally, run `decode.py` to get the ROUGE scores.

### Post-Processing

 - Input file has the same format as standard `result.txt` that looks like below snippet:
```
x: ~~~~
y: ~~~~
y_pred: (certain prediction)
```
 - Output file will be of same format as that of input file, only with changes in `y_pred` parts.
```
x: ~~~~
y: ~~~~
y_pred: (changed contents based on rules)
```
- Usage

First, set `input_result_path` and `output_result_path` in `post_process.py`

Then run :

```
python post_process.py
```

 - Functionality

   - Replace strange words based on dictionary.
   - Tag [주의] to predictions that possibly need human correction.
