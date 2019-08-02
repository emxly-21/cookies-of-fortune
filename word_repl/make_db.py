import node
from node import Node
import csv
import numpy as np
import gensim
from gensim.models.keyedvectors import KeyedVectors
import pickle

def make_data(pathGl, pathFr, cutoff = .8, num_reps = 10000000):
    '''
    Creates a database of all
    components of the same color
    in the database of words
    (utilizing the Whispers
    algorithm), mapping to the
    most common word in that
    component. Also puts that
    component into a pickle file.

    Parameters:
    -----------
    pathGl: String
    A string containing a path to the
    GloVe file of 50-dimensional
    embeddings.

    pathFr: String
    A string containing a path to the
    csv file of words by frequency.

    cutoff: float
    A float representing the lowest
    cosine similarity for two words
    to be considered connected.

    num_reps: int
    An integer representing the
    number of interations for which
    the Whispers algorithm should run.

    Returns:
    --------
    Dictionary{String : String}
    A dictionary that maps words
    to the most common words that
    are in the same component.
    Also puts this dictionary
    into a pickle file.
    '''

    glove = KeyedVectors.load_word2vec_format(pathGl, binary=False)

    print("glove loaded")

    freqcnt = {}
    with open(pathFr, newline='\n') as file:
        reader = csv.reader(file)
        ctr = 0
        for row in reader:
            word = row[0]
            if(ctr != 0):
                freqcnt[word] = ctr
            ctr = ctr+1
    
    print("freq loaded")
    
    ree = {a:b for a,b in zip(glove.vocab, range(len(glove.vocab)))}
    ree2 = {a:b for b,a in ree.items()}

    print("starting node making")

    nodes = []
    for word in glove.vocab:
        mc = glove.most_similar(word) # raise this if we need more connections
        near = []
        for word, sim in mc:
            if sim < cutoff:
                break
            near.append(ree[word])
        cn = Node(ree[word], near, glove[word])
        nodes.append(cn)
        if(len(nodes) % 10000 == 0):
            print("loaded", len(nodes))

    print("starting whispers")
    
    nodes = node.whispers(nodes, numTimes = num_reps) # idk if this is enough oop
    rev_dict = {} # number -> all words with that number
    for cn in nodes:
        val = cn.label
        if val not in rev_dict:
            rev_dict[val] = []
        else:
            rev_dict[val].append(ree2[cn])
    
    print("done with whispers")

    fin = {}
    for val in rev_dict:
        most_common = ""
        val = 1000000
        for wrd in rev_dict[val]:
            if(wrd in freqcnt and freqcnt[wrd] < val):
                most_common = wrd
                val = freqcnt[wrd]
        
        for wrd in rev_dict[val]:
            fin[wrd] = most_common
    
    print("about to pickle")

    pickle_out = open("database.pickle", "wb")
    pickle.dump(fin, pickle_out)
    pickle_out.close()



def load_data(pathGl, pathFr, cutoff=.85):
    '''
    Creates a database of all
    components of the same color
    in the database of words
    (utilizing the Whispers
    algorithm), mapping to the
    most common word in that
    component. Also puts that
    component into a pickle file.

    Parameters:
    -----------
    pathGl: String
    A string containing a path to the
    GloVe file of 50-dimensional
    embeddings.

    pathFr: String
    A string containing a path to the
    csv file of words by frequency.

    cutoff: float
    A float representing the lowest
    cosine similarity for two words
    to be considered connected.

    Notes:
    ------
    Tuple(Dictionary{String : int}, Dictionary{int: String}, Dictionary{String : List[String]})
    A dictionary that maps words
    to the most common words that
    are in the same component.
    This dictionary is put into a
    pickle file, named "info.pickle".
    '''

    glove = KeyedVectors.load_word2vec_format(pathGl, binary=False)
    print("glove loaded")

    freqcnt = {}
    with open(pathFr, newline='\n') as file:
        reader = csv.reader(file)
        ctr = 0
        for row in reader:
            word = row[0]
            if(ctr != 0):
                freqcnt[word] = ctr
            ctr = ctr+1
    print("freq loaded")

    with open ("./dat/stopwords.txt", 'r') as fil:
        stops = set()
        for line in fil:
            stops.update([i.strip() for i in line.split('\t')])
    print("stops loaded")

    ree = {}
    ree2 = {}

    cnt = 0
    for wrd in glove.vocab:
        if wrd not in stops and wrd.isalpha() and len(wrd)>0:
            # print("hi")
            ree[wrd] = cnt
            ree2[cnt] = wrd
            cnt += 1
    
    # ree = {a:b for a,b in zip(glove.vocab, range(len(glove.vocab)))}
    # ree2 = {a:b for b,a in ree.items()}

    print("starting node making")

    nodes = []
    for word in glove.vocab:
        if word in stops or not word.isalpha() or len(word)==0:
            continue
        
        mc = glove.most_similar(word, topn=10) # raise this if we need more connections
        near = []
        for wrd, sim in mc:
            if sim < cutoff:
                break
            if wrd not in stops and wrd.isalpha() and len(wrd)>0:
                near.append(ree[wrd])
        nodes.append(near)
        if(len(nodes) % 10000 == 0):
            print("loaded", len(nodes))
    
    res = (ree, ree2, nodes)
    pickle_out = open("info.pickle", "wb")
    pickle.dump(res, pickle_out)
    pickle_out.close()

def make_pic(pathGl, pathFr, num_reps=10000000):
    '''
    Creates a database mapping each word to a more common word that is similar to it.

    This function utilizes the Whispers algorithm, along with a pre-created pickle file
    containing an adjacency list for the graph, 

    Parameters:
    -----------
    pathGl: String
    A string containing a path to the
    GloVe file of 50-dimensional
    embeddings.

    pathFr: String
    A string containing a path to the
    csv file of words by frequency.

    num_reps: integer
    The number of iterations that the
    Whispers algorithm should run for.

    Notes:
    A pickle file named "database.pickle"
    is created. This stores the dictionary,
    which is a Dictionary{String : String}.
    '''
    glove = KeyedVectors.load_word2vec_format(pathGl, binary=False)
    print("glove loaded")

    freqcnt = {}
    with open(pathFr, newline='\n') as file:
        reader = csv.reader(file)
        ctr = 0
        for row in reader:
            word = row[0]
            if(ctr != 0):
                freqcnt[word] = ctr
            ctr = ctr+1
    print("freq loaded")

    with open ("./dat/stopwords.txt", 'r') as fil:
        stops = set()
        for line in fil:
            stops.update([i.strip() for i in line.split('\t')])
    print("stops loaded")

    pickleName="info.pickle"
    with open(pickleName,"rb") as infoFile:
        tup=pickle.load(infoFile)
    forw = tup[0]
    back = tup[1]
    adj = tup[2]
    nodes = []
    print("pickle loaded")
    for word in glove.vocab:
        if(word in stops or not word.isalpha() or len(word) == 0):
            continue
        cn = Node(forw[word], adj[forw[word]], glove[word])
        nodes.append(cn)
    
    print("starting whispers")
    
    nodes = node.whispers(nodes, numTimes = num_reps) # idk if this is enough oop
    rev_dict = {} # number -> all words with that number
    for cn in nodes:
        val = cn.label
        if val not in rev_dict:
            rev_dict[val] = []
        else:
            bleh = back[cn.id]
            rev_dict[val].append(bleh)
    
    print("done with whispers")

    fin = {}
    for val in rev_dict:
        most_common = ""
        mn = 1000000
        for wrd in rev_dict[val]:
            if(wrd in freqcnt and freqcnt[wrd] < mn):
                most_common = wrd
                mn = freqcnt[wrd]
        
        for wrd in rev_dict[val]:
            fin[wrd] = most_common
    
    print("about to pickle")

    pickle_out = open("database.pickle", "wb")
    pickle.dump(fin, pickle_out)
    pickle_out.close()