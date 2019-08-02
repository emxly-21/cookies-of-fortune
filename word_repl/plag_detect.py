import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models.keyedvectors import KeyedVectors
import re, string

punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
def strip_punc(corpus):
    """ Removes all punctuation from a string.

        Parameters
        ----------
        corpus : str

        Returns
        -------
        str
            the corpus with all punctuation removed"""
    # substitute all punctuation marks with ""
    return punc_regex.sub('', corpus)

def main(pathGl):
    glove = KeyedVectors.load_word2vec_format(pathGl, binary=False)
    print("glove loaded")
    inp1 = input("enter first text\n")
    inp2 = input("enter second text\n")
    words1 = strip_punc(inp1).split()
    words2 = strip_punc(inp2).split()

    vecs1 = np.array([glove[word] for word in words1])
    vecs2 = np.array([glove[word] for word in words2])
    print(vecs1.shape)

    tot = 0.

    # make smthn that gets called
    for i in range(len(vecs1)):
        res = cosine_similarity(vecs1[i].reshape(1, -1), vecs2[max(i-5, 0):min(i+5, len(vecs2))].reshape(-1, len(vecs2[0])))
        print(res)
        tot += max(res[0])

    print(tot/len(vecs1))