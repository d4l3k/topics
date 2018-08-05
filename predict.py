import numpy as np

import model
import vectors

lt, ld = model.load_trained()

def sentence_embedding(sentence):
    return sentence_embeddings([sentence])

def sentence_embeddings(sentences):
    return lt.predict(np.array([
        vectors.weights_arr(vectors.split_sentence(sentence))
        for sentence in sentences
    ]))

def topic_embedding(topic):
    return topic_embeddings([topic])

def topic_embeddings(topics):
    return ld.predict(np.array([
        vectors.weights(topic) for topic in topics
    ]))

if __name__ == "__main__":
    s = sentence_embedding('Hello, how are you?').flatten()
    cows = topic_embedding('cows').flatten()
    greeting = topic_embedding('greeting').flatten()
    hello = topic_embedding('hello').flatten()
    print(np.dot(s, cows))
    print(np.dot(s, greeting))
    print(np.dot(s, hello))
