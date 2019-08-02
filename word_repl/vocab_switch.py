import pickle
import gensim
from gensim.models.keyedvectors import KeyedVectors

def main():
    '''
    
    '''
    pickleName="database.pickle"
    with open(pickleName,"rb") as dictFile:
        dic=pickle.load(dictFile)

    inp = input("Please input text:\n")
    words = inp.split()
    for i in range(len(words)):
        if words[i] in dic:
            words[i] = dic[words[i]]
    oup = " ".join(words)
    print(oup)

def find_bad():
    pickleName="database.pickle"
    with open(pickleName,"rb") as dictFile:
        dic=pickle.load(dictFile)
    
    for wrd in dic:
        res = dic[wrd]
        if(wrd != res):
            print(wrd, res)
    
    print("hmm")