import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D


class CNNModel:
    @staticmethod
    def reshape_data(data: np.array) -> np.array:
        num_samples = data.shape[0]
        return data.reshape(num_samples, *data[0].shape, 1)

    @staticmethod
    def create(no_classes, input_shape=(40, 174, 1)):
        model = Sequential(name='CNN')
        model.add(Conv2D(filters=16, kernel_size=2, input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(GlobalAveragePooling2D())

        model.add(Dense(no_classes, activation='softmax'))

        return model
