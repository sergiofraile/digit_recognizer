import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


def prepare_data(num_classes, train_test_ratio = 0.8):
    # Load the train data and split it .8/.2

    train_data = pd.read_csv("./data/train.csv")

    print('train_data.shape=', train_data.shape)

    train_size = int(len(train_data) * train_test_ratio)
    y_train = train_data['label'][:train_size]
    y_test = train_data['label'][train_size:]

    x_train = train_data.drop(labels=['label'], axis=1)[:train_size]
    x_test = train_data.drop(labels=['label'], axis=1)[train_size:]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print(x_train.shape)

    x_train = x_train.values.reshape(-1, 28, 28, 1)
    x_test = x_test.values.reshape(-1, 28, 28, 1)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print('x_train.shape=', x_train.shape)
    print('y_train.shape=', y_train.shape)

    print('x_test.shape=', x_test.shape)
    print('y_test.shape=', y_test.shape)

    return x_train, y_train, x_test, y_test


def fit_model(m, kx_train, ky_train, kx_test, ky_test, batch_size=128, max_epochs=1000):

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc',
                                                   min_delta=0.01,
                                                   patience=10,
                                                   verbose=1,
                                                   mode='max')

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                                  factor=0.5,
                                                  patience=10,
                                                  min_lr=0.0001,
                                                  verbose=1)

    m.fit(kx_train,
          ky_train,
          batch_size=batch_size,
          epochs=max_epochs,
          verbose=1,
          validation_data=(kx_test, ky_test),
          callbacks=[early_stopping, reduce_lr])


def fit_model_with_geneartor(m, x_train, y_train, x_test, y_test, batch_size=128, max_epochs=1000):

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc',
                                                   min_delta=0.01,
                                                   patience=10,
                                                   verbose=1,
                                                   mode='max')

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                                  factor=0.5,
                                                  patience=10,
                                                  min_lr=0.0001,
                                                  verbose=1)

    gen = ImageDataGenerator(rotation_range=8,
                             width_shift_range=0.08,
                             shear_range=0.3,
                             height_shift_range=0.08,
                             zoom_range=0.08)

    test_gen = ImageDataGenerator()

    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
    test_generator = test_gen.flow(x_test, y_test, batch_size=batch_size)

    m.fit_generator(train_generator,
                    steps_per_epoch=x_train.shape[0] / batch_size,
                    epochs=max_epochs,
                    validation_data=test_generator,
                    validation_steps=x_test.shape[0] / batch_size,
                    callbacks=[early_stopping, reduce_lr])


def predict_results(m):
    test_data = pd.read_csv("./data/test.csv")
    test_data = test_data.astype('float32')
    test_data /= 255
    test_data = test_data.values.reshape(-1, 28, 28, 1)

    return m.predict(test_data)


batch_size = 128
num_classes = 10
epochs = 20
train_test_ratio = 0.8


# input image dimensions
img_rows, img_cols = 28, 28


# Load and preparing the train and test data
x_train, y_train, x_test, y_test = prepare_data(num_classes, train_test_ratio)


# Creating the model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Data augmentation

gen = ImageDataGenerator(rotation_range=8,
                         width_shift_range=0.08,
                         shear_range=0.3,
                         height_shift_range=0.08,
                         zoom_range=0.08)

test_gen = ImageDataGenerator()

train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
test_generator = test_gen.flow(x_test, y_test, batch_size=batch_size)


# Fitting the model
# fit_model(model, x_train, y_train, x_test, y_test, batch_size, epochs)
fit_model_with_geneartor(model, x_train, y_train, x_test, y_test, batch_size, epochs)

# Predict the test results
results = predict_results(model)
results = np.argmax(results, axis=1)
results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
submission.to_csv("submission.csv", index=False)
