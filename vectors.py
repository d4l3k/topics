import io
import numpy as np
from tqdm import tqdm
import re

import os
#os.environ['LMDB_FORCE_CFFI'] = '1'

import lmdb

vector_file = 'wiki-news-300d-1M.vec'

db = lmdb.open('vectors.lmdb', map_size=10000000000, max_readers=100)

def load_words(fname=vector_file):
    out = []
    with io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split())
        for line in tqdm(fin, total=n, desc='load_words'):
            tokens = line.split(' ', 1)
            key = tokens[0]
            out.append(key)
    return out

def load_vectors(fname=vector_file):
    with io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split())
        with db.begin(write=True) as txn:
            for line in tqdm(fin, total=n, desc='load_vectors'):
                tokens = line.rstrip().split(' ')
                key = tokens[0]
                values = np.array(tokens[1:]).astype(np.float)
                txn.put(key.encode(), values.tobytes())

def weights(word):
    return weights_arr([word])[0]

def weights_arr(words, dim=300):
    out = []
    with db.begin(buffers=True) as txn:
        for word in words:
            buf = txn.get(word.lower().encode())
            if buf is None:
                out.append(np.zeros(dim))
            else:
                out.append(np.frombuffer(buf, dtype=np.float))

    return out

def split_sentence(s):
    return re.findall(r"[\w]+|[.,!?;'-]", s)

if __name__ == "__main__":
    #load_vectors('wiki-news-300d-1M.vec')
    print(load_words()[100])
    #print(weights_arr(split_sentence("Hello! How are you're yes woof., meh")))
