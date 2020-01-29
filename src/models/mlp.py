import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation, Dropout


class MLPModel:
    @staticmethod
    def reshape_data(data):
        """Receive numpy array with features as 3D array, where dimensions are (#samples, #mfcc, #frames)"""
        # axis=2: calculate mean across all frames per sample and mfcc
        x = np.mean(data, axis=2)
        return x

    # @staticmethod
    # def create(no_classes, input_shape=(40,)):
    #     model = Sequential(name='MLP')
    #     model.add(Dense(256, input_shape=input_shape))
    #     model.add(Activation('relu'))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(256))
    #     model.add(Activation('relu'))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(no_classes))
    #     model.add(Activation('softmax'))
    #
    #     return model

    @staticmethod
    def create(no_classes, input_shape=(40,)):
        model = Sequential(name='MLP')
        model.add(Dense(512, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.25))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(no_classes, activation='softmax'))

        return model
