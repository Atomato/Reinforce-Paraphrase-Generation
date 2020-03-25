from evaluator.Model import Decomposable
from evaluator import Config
from evaluator.Utils import *
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
import tensorflow as tf

class Evaluator(object):
    def __init__(self):
        # read config
        config = Config.ModelConfig()
        arg = config.arg

        self.vocab_dict = load_vocab(arg.vocab_path)
        embeddings = load_embeddings(arg.embedding_path, self.vocab_dict)

        arg.n_vocab, arg.embedding_size = embeddings.shape
        arg.n_classes = len(CATEGORIE_ID)

        self.model = Decomposable(arg.seq_length, arg.n_vocab, arg.embedding_size, \
                            arg.hidden_size, arg.attention_size, arg.n_classes, \
                            arg.batch_size, arg.learning_rate, arg.optimizer, \
                            arg.l2, arg.clip_value)

        self.best_path = arg.best_path
        self.sess = tf.Session()
        saver = tf.train.Saver(max_to_keep=5)
        saver.restore(self.sess, self.best_path)

    def sentence2Index(self, encode_sent, decode_sent, maxLen = 100):
        """
        :return: s1Pad: padded sentence1
                 s2Pad: padded sentence2
                 s1Mask: actual length of sentence1
                 s2Mask: actual length of sentence2
        """
        vocabDict = self.vocab_dict

        s1List, s2List, labelList = [], [], []
        s1Mask, s2Mask = [], []
        try:
            s1 = [v.strip() for v in encode_sent.strip().split()]
            s2 = [v.strip() for v in decode_sent.strip().split()]
            if len(s1) > maxLen:
                s1 = s1[:maxLen]
            if len(s2) > maxLen:
                s2 = s2[:maxLen]
            s1List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s1])
            s2List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s2])
            s1Mask.append(len(s1))
            s2Mask.append(len(s2))
        except:
            ValueError('Input Data Value Error!')

        s1Pad, s2Pad = pad_sequences(s1List, maxLen, padding='post'), pad_sequences(s2List, maxLen, padding='post')
        s1MaskList, s2MaskList = (s1Pad > 0).astype(np.int32), (s2Pad > 0).astype(np.int32)
        enc = OneHotEncoder(sparse=False)
        s1Mask = np.asarray(s1Mask, np.int32)
        s2Mask = np.asarray(s2Mask, np.int32)

        return s1Pad, s1MaskList, s2Pad, s2MaskList

    def reward(self, encode_sent, decode_sent):
        premise, premise_mask, hypothesis, hypothesis_mask = \
                                            self.sentence2Index(encode_sent, decode_sent)

        feed_dict = {self.model.premise: premise,
                     self.model.premise_mask: premise_mask,
                     self.model.hypothesis: hypothesis,
                     self.model.hypothesis_mask: hypothesis_mask,
                     self.model.dropout_keep_prob: 1.0}
        logits = self.sess.run([self.model.logits], feed_dict = feed_dict)
        logits = np.array(logits)
        logits = logits.reshape(-1)

        return float(softmax(logits)[CATEGORIE_ID['entailment']])

    def close(self):
        self.sess.close()

