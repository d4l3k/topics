import predict
import vectors

import faiss
from tqdm import tqdm
import numpy as np
import math

class Batch():
    def __init__(self, iterator, size=1):
        self.iterator = iterator
        self.size = size

    def __getitem__(self, i):
        batch = self.iterator[i*self.size:(i+1)*self.size]
        if len(batch) == 0:
            raise StopIteration
        return batch

    def __len__(self):
        return math.ceil(len(self.iterator)/self.size)

class Index():
    def __init__(self):
        self.words = vectors.load_words()
        self.index = faiss.IndexFlatIP(vectors.DIMS)
        for words in tqdm(Batch(self.words, size=1024), desc="build_index"):
            embeddings = predict.topic_embeddings(words)
            self.index.add(embeddings)

    def find(self, sentence, k=30):
        embedding = predict.sentence_embedding(sentence)
        print('embedding {}'.format(embedding))
        scores, indexes = self.index.search(embedding, k)
        topics = [
            self.words[i] for i in indexes[0]
        ]
        topic_embeddings = predict.topic_embeddings(topics)
        print(np.dot(topic_embeddings, np.transpose(embedding)))
        return topics, scores, indexes


if __name__ == "__main__":
    index = Index()
    print(index.find('The food was really good!'))
    print(index.find('Oceania was at war with Eastasia: Oceania had always been at war with Eastasia.'))

