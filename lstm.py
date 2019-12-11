'''
VAE-LSTM by Yuning Wu

BI-LSTM by Rohith Pillai

Parameter tuning and interface implementation

Code reference: DeepJazz (Baseline)
Author:     Ji-Sung Kim
Project:    deepjazz
Purpose:    Generate jazz using a deep learning model (lstm in deepjazz).

Some code adapted from Evan Chow's jazzml, https://github.com/evancchow/jazzml 
with express permission.

Code was built while significantly referencing public examples from the
Keras documentation on GitHub:
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python generator.py [# of epochs]

    Note: running Keras/Theano on GPU is formally supported for only NVIDIA cards (CUDA backend).
'''
from __future__ import print_function

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector, Bidirectional
from keras.layers.core import Dense, Activation, Dropout, Flatten, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras.regularizers import l1, l2
from keras import objectives
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

''' Build a 2-layer LSTM from a training corpus '''
def build_model(corpus, val_indices, max_len, N_epochs=128, model_choice=None):
    # number of different values or words in corpus
    N_values = len(set(corpus))

    # cut the corpus into semi-redundant sequences of max_len values
    step = 3
    sentences = []
    next_values = []
    for i in range(0, len(corpus) - max_len, step):
        sentences.append(corpus[i: i + max_len])
        next_values.append(corpus[i + max_len])
    print('nb sequences:', len(sentences))

    # transform data into binary matrices
    X = np.zeros((len(sentences), max_len, N_values), dtype=np.bool)
    y = np.zeros((len(sentences), N_values), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, val in enumerate(sentence):
            X[i, t, val_indices[val]] = 1
        y[i, val_indices[next_values[i]]] = 1
    
    
    if model_choice=='lstm':
        # build a 2 stacked LSTM
        # default
        model = lstm(input_dim=N_values, timesteps=max_len)
        plot_model(model, to_file='lstm.png')
        history = model.fit(X, y, batch_size=128, epochs=N_epochs)
    
    elif model_choice=='vae-lstm':
        model, _, _ = vae_lstm(input_dim=N_values, 
                               timesteps=max_len, 
                               batch_size=N_epochs,
                               middle_dim=128,
                               latent_dim=64, 
                               epsilon_std=0.5) 
        plot_model(model, to_file='vae-lstm.png')
        history = model.fit(X, X, batch_size=128, epochs=N_epochs)

    elif model_choice=='bi-lstm':
        model = bi_lstm(input_dim=N_values, timesteps=max_len)
        plot_model(model, to_file='bi-lstm.png')
        history = model.fit(X, y, batch_size=128, epochs=N_epochs)

    else:
        raise Exception

    # Plot training & validation accuracy values
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    # Plot training & validation loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    return model


def lstm(input_dim, timesteps):
    # build a 2 stacked LSTM
    # default
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(timesteps, input_dim)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(input_dim))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def bi_lstm(input_dim, timesteps):
    # build a stacked Bi-LSTM
    model = Sequential() #1
    model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(timesteps, input_dim))))
    model.add(Dropout(0.2))
    #2
    model.add(Bidirectional(LSTM(610, return_sequences=True)))
    model.add(Dropout(0.2))
    #3
    model.add(Bidirectional(LSTM(610, return_sequences=True)))
    model.add(Dropout(0.2))
    #4
    model.add(Bidirectional(LSTM(610, return_sequences=True)))
    model.add(Dropout(0.2))
    #4
    model.add(Bidirectional(LSTM(610, return_sequences=True)))
    model.add(Dropout(0.2))
    #5
    model.add(Bidirectional(LSTM(610, return_sequences=True)))
    model.add(Dropout(0.2))
    #6
    model.add(Bidirectional(LSTM(610, return_sequences=True)))
    model.add(Dropout(0.2))
    #7
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.2))

    model.add(Dense(input_dim, kernel_regularizer=l1(1e-3)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', )
    # model.fit(X, y, batch_size=128, nb_epoch=N_epochs)
    return model
    


def vae_lstm(input_dim, timesteps, batch_size, middle_dim, latent_dim, epsilon_std=1.):

    """
    References
        - [Keras LSTM VAE] https://github.com/twairball/keras_lstm_vae/blob/master/lstm_vae/vae.py
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    def sampling(args):
        z_mean, z_sigma = args
        epsilon = K.random_normal(shape=(timesteps, latent_dim, ), mean=0., stddev=epsilon_std)
        z = K.mean(z_mean + z_sigma * epsilon, axis=1)
        print('batch size', batch_size)
        print('input dim', input_dim)
        print('timesteps', timesteps)
        return z
    
    def vae_loss(x, x_decoded):
        x_loss = objectives.mse(x, x_decoded)
        kl_loss = - 0.5 * K.mean(1 + z_sigma - K.square(z_mean) - K.exp(z_sigma), axis=-1)
        return x_loss + kl_loss

    # encoding
    x = Input(shape=(timesteps, input_dim, ))
    h = LSTM(middle_dim, return_sequences=True)(x)

    # z
    z_mean = Dense(latent_dim)(h)
    z_sigma = Dense(latent_dim)(h)
    print('This is mu:', K.int_shape(z_mean), 'latent_dim', latent_dim)

    z = Lambda(sampling, output_shape=(latent_dim, ))([z_mean, z_sigma])
    
    # decoding
    decoder_h = LSTM(middle_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True)

    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    # encoder
    x_decoded = decoder_mean(h_decoded)  
    encoder = Model(x, z_mean)
    encoder.summary()

    # decoder
    decoder_input = Input(shape=(latent_dim, ))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)
    _x_decoded = decoder_mean(_h_decoded)

    decoder = Model(decoder_input, _x_decoded)
    decoder.summary()

    vae = Model(x, x_decoded)

    vae.compile(optimizer='adam', loss=vae_loss)
    # vae.compile(optimizer='rmsprop', loss=vae_loss)
    
    return vae, encoder, decoder