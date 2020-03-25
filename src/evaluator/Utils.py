import pickle
import numpy as np

UNKNOWN = '<UNK>'
PADDING = '<PAD>'
EQ = '<EQ>'
CATEGORIE_ID = {'contradiction' : 0, 'entailment' : 1}

# init embeddings randomly
def init_embeddings(vocab, embedding_dims):
    """
    :param vocab: word nums of the vocabulary
    :param embedding_dims: dimension of embedding vector
    :return: randomly init embeddings with shape (vocab, embedding_dims)
    """
    rng = np.random.RandomState(None)
    random_init_embeddings = rng.normal(size = (len(vocab), embedding_dims))
    return random_init_embeddings.astype(np.float32)

# print tensor shape
def print_shape(varname, var):
    """
    :param varname: tensor name
    :param var: tensor variable
    """
    print('{0} : {1}'.format(varname, var.get_shape()))

# load vocabulary
def load_vocab(vocabPath, threshold = 0):
    """
    :param vocabPath: path of vocabulary file
    :param threshold: mininum occurence of vocabulary, if a word occurence less than threshold, discard it
    :return: vocab: vocabulary dict {word : index}
    """
    vocab = {}
    index = 2
    vocab[PADDING] = 0
    vocab[UNKNOWN] = 1
    vocab[EQ] = 2
    with open(vocabPath, encoding='utf-8') as f:
        for line in f:
            items = [v.strip() for v in line.split('||')]
            if len(items) != 2:
                print('Wrong format: ', line)
                continue
            word, freq = items[0], int(items[1])
            if freq >= threshold:
                vocab[word] = index
                index += 1
    return vocab

# load pre-trained embeddings
def load_embeddings(path, vocab):
    """
    :param path: path of the pre-trained embeddings file
    :param vocab: word nums of the vocabulary
    :return: pre-trained embeddings with shape (vocab, embedding_dims)
    """
    with open(path, 'rb') as fin:
        _embeddings, _vocab = pickle.load(fin)
    embedding_dims = _embeddings.shape[1]
    embeddings = init_embeddings(vocab, embedding_dims)
    for word, id in vocab.items():
        if word in _vocab:
            embeddings[id] = _embeddings[_vocab[word]]
    return embeddings.astype(np.float32)