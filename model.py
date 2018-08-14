import io
import json
import random

import numpy as np
from tqdm import tqdm

import keras.utils
from keras.models import Sequential, Model, load_model
from keras.layers import (
    Dense, Dropout, Input, Embedding, LSTM, GRU, maximum, Dot, add, subtract,
    Lambda,
)
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import keras.losses

import vectors

DIMS = vectors.DIMS
BATCH_SIZE = 100
MODEL_NAME = 'weights.hdf5'

def load_trained():
    model = create_model()
    model.load_weights(MODEL_NAME)

    lt = Model(inputs=model.get_layer('g').input,
            outputs=model.get_layer('lt').output)
    ld = Model(inputs=model.get_layer('dpos').input,
            outputs=model.get_layer('ld').get_output_at(0))
    return lt, ld

def normalize(weights):
    return weights/K.sum(weights, axis=1, keepdims=True)

def create_model():
    g = Input(shape=(DIMS,), dtype=np.float, name='g')
    #lstm = LSTM(300)(g)
    #intermediate = Dense(DIMS)(g)
    lt = Dense(DIMS, activation='sigmoid', name='lt')(g)
    #lt = Lambda(normalize, name='lt')(lt_raw)

    dpos = Input(shape=(DIMS,), dtype=np.float, name='dpos')
    dneg = Input(shape=(DIMS,), dtype=np.float, name='dneg')

    ld = Dense(DIMS, activation='softmax', name='ld')

    ldpos = ld(dpos)
    ldneg = ld(dneg)

    dot = Dot(axes=1)
    lspos = dot([lt, ldpos])
    lsneg = dot([lt, ldneg])

    omega = Input(tensor=K.repeat_elements(K.constant([[1.0]]), BATCH_SIZE, 0), name='omega')
    zero = Input(tensor=K.repeat_elements(K.constant([[0.0]]), BATCH_SIZE, 0), name='zero')

    cost = maximum([
        zero,
        add([lsneg, subtract([omega, lspos])])
    ], name='val_loss')

    #def loss(y_true, y_pred):
    #    return y_pred

    model = Model(inputs=[g, dpos, dneg, omega, zero], outputs=[cost])

    #optimizer = optimizers.Adam()
    # GRU, loss = 1199
    #optimizer = optimizers.Adam(lr=0.0001)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'acc'])
    print(model.summary())

    return model

example_dataset_name = 'yelp_academic_dataset_review.json'

def build_example_index(fname=example_dataset_name):
    indexes = []
    with tqdm(desc='build_example_index') as pbar:
        with io.open(fname, 'r', newline='\n') as fin:
            while len(fin.readline()) >= 4:
                indexes.append(fin.tell())
                pbar.update()

    return np.array(indexes)

def fetch_examples(indexes, fname=example_dataset_name):
    out = []
    with io.open(fname, 'r', newline='\n') as fin:
        for i in indexes:
            fin.seek(i)
            line = fin.readline()
            try:
                data = json.loads(line)
            except Exception as e:
                print('read json error: {}, {}\n{}'.format(i,line, e))
                # TODO: figure out why this is happening
                fin.seek(0)
                line = fin.readline()
                data = json.loads(line)

            out.append(data["text"])

    return out

def get_negative_word(all_words, neg_words):
    for _ in range(100):
        word = random.choice(all_words)
        if not word in neg_words:
            return word

    raise "couldn't find valid word"

class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=BATCH_SIZE, shuffle=True, dims=DIMS):
        self.dims = dims
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = build_example_index()
        self.words = vectors.load_words()

        self.on_epoch_end()

    def __len__(self):
        return min(int(np.floor(len(self.indexes) / self.batch_size)), 10000000)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        examples = fetch_examples(indexes)
        sentences = [vectors.split_sentence(s) for s in examples]
        #max_len = max([len(s) for s in sentences])
        #weights = np.zeros((len(indexes), max_len, self.dims))
        weights = np.zeros((len(indexes), self.dims))
        dpos = np.zeros((len(indexes), self.dims))
        dneg = np.zeros((len(indexes), self.dims))


        # Generate data
        for i, example in enumerate(examples):
            words = sentences[i]
            eweights = vectors.weights_arr(words)
            weights[i, :] = eweights
            dpos[i, :] = random.choice(eweights)
            dneg[i, :] = vectors.weights(get_negative_word(self.words, words))

        return {
            'g': weights,
            'dpos': dpos,
            'dneg': dneg
        }, np.zeros(len(indexes))

def train_model(model):
    training_generator = DataGenerator()
    checkpointer = ModelCheckpoint(
        filepath=MODEL_NAME,
        verbose=1,
        save_best_only=True,
        monitor='loss',
    )
    model.fit_generator(generator=training_generator,
            epochs=100,
            callbacks=[checkpointer],
            #validation_data=validation_generator,
            use_multiprocessing=True,
            workers=8)

if __name__ == "__main__":
    model = create_model()
    plot_model(model, to_file='model.png', show_shapes=True)
    train_model(model)
