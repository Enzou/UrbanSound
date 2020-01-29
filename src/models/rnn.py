import numpy as np
from keras import Sequential
from keras.layers import LeakyReLU
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM as LSTMLayer
from keras.layers import SimpleRNN as RNNLayer
from keras.layers import GRU as GRULayer


class RNNModel:
    # "RNNs are tricky. Choice of batch size is important, choice of loss and optimizer is critical, etc. Some configurations won't converge." - https://keras.io/examples/imdb_lstm/

    @staticmethod
    def reshape_data(data: np.array):
        """Receive numpy array with features as 3D array, where dimensions are (#samples, #mfcc, #frames)"""

        # old reshape
        return data.transpose((0, 2, 1))   # switch position of #mfcc and #frames, because #mfcc is size of vector per timestep

        # # CNN reshape
        # num_samples = data.shape[0]
        # return data.reshape(num_samples, *data[0].shape)

    @staticmethod
    def create(no_classes, input_shape=(174, 40)):

        print(f"Input shape: {input_shape}")
        model = Sequential(name="RNN")
        # model.add(LeakyReLU(alpha=0.1))    # https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
        # model.add(TimeDistributed(Dense(no_classes, activation='softmax')))
        # model.add(RNNLayer(32, return_sequences=True, activation='relu', input_shape=input_shape, dropout=0.2))
        model.add(RNNLayer(174, return_sequences=False, activation='relu', dropout=0.3, input_shape=input_shape))
        # model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(no_classes, activation='softmax'))


        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model


class LSTMModel(RNNModel):

    @staticmethod
    def create(no_classes, input_shape=(174, 40)):
        print(f"Input shape: {input_shape}")
        model = Sequential(name="LSTM")
        # model.add(LSTMLayer(40, return_sequences=False, activation='relu', input_shape=input_shape, dropout=0.2))
        # model.add(LeakyReLU(alpha=0.1))    # https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
        # model.add(Dense(no_classes, activation='softmax'))

        model.add(LSTMLayer(174, input_shape=input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(no_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model


class GRUModel(RNNModel):
    @staticmethod
    def create(no_classes, input_shape=(174, 40)):
        model = Sequential(name="GRU")

        model.add(GRULayer(174, input_shape=input_shape))
        model.add(Dense(128, activation='relu'))

        # model with returned sequences is hardly better
        # model.add(GRULayer(40, return_sequences=True, activation='relu', input_shape=input_shape, dropout=0.2))
        # model.add(GRULayer(40, return_sequences=False, activation='relu', input_shape=input_shape, dropout=0.2))
        model.add(Dense(no_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        return model
