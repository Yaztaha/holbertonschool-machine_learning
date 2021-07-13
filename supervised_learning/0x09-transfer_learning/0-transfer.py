#!/usr/bin/env python3
""" Transfer learning """

import numpy as np
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    pre-processes the data for your model:
    - X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10
    data, where m is the number of data points
    - Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns: X_p, Y_p
        - X_p: numpy.ndarray containing the preprocessed X
        Y_p: numpy.ndarray containing the preprocessed Y
    """
    X = K.applications.inception_v3.preprocess_input(X)
    Y = K.utils.to_categorical(Y, 10)

    return X, Y


if __name__ == '__main__':
    """
    Testing inceptionV3 model with transfer learning on CIFAR10 dataset
    """

    input_tensor = K.Input(shape=(32, 32, 3))
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    x_train = np.concatenate((x_train, np.flip(x_train, 2)), 0)
    y_train = np.concatenate((y_train, y_train), 0)

    input_tensor_resize = K.layers.Lambda(
        lambda image: K.backend.resize_images(
            image, (int(100 / 32)), (int(100 / 32)),
            "channels_last")
    )(input_tensor)

    y = K.layers.ZeroPadding2D(padding=4)(input_tensor_resize)

    pre_trained_model = K.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_tensor=y,
    )

    x = K.layers.Flatten()(pre_trained_model.output)
    x = K.layers.Dense(1024, activation='relu')(x)
    x = K.layers.Dense(512, activation='relu')(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.2)(x)

    x = K.layers.Dense(10, activation='softmax')(x)

    model = K.models.Model(pre_trained_model.input, x)

    lrr = K.callbacks.ReduceLROnPlateau(
        monitor='val_acc',
        factor=.01,
        patience=3,
        min_lr=1e-5)

    es = K.callbacks.EarlyStopping(monitor='val_acc',
                                   mode='max',
                                   verbose=1,
                                   patience=10)

    mc = K.callbacks.ModelCheckpoint('cifar10.h5',
                                     monitor='val_acc',
                                     mode='max',
                                     verbose=1,
                                     save_best_only=True)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=128,
                        callbacks=[es, mc, lrr],
                        epochs=20,
                        verbose=1)

    model.save('cifar10.h5')
