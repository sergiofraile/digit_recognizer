import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from utils import *

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
