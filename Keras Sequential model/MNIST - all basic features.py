# Best run: 0.9918

#to try:
#   Use all 60000 images, do more epochs
#   Filters: different numbers, sizes, steps
#   Pool size
#   Dropout proportion
#   Size of dense layers
#   Regularization
#   Learning rate (meaning of decay, momentum, nesterov?)
#   Loss function and optimizer

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import random

nb_classes = 10

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Format images, and scale inputs to 0-1 rather than 0-255
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Make labels have one-hot encoding
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# Make the model
model = Sequential()
#general Conv2D layer: model.add(Conv2D(filters, kernel_size, strides=(1, 1), activation=None, bias_regularizer=None))
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Make lists of which test images were correct and incorrect
predicted_classes = model.predict_classes(x_test) #predict_classes outputs the class chosen by the trained model
y_test = np.argmax(y_test, axis=1) #undo one-hot encoding on y_test, so we can compare
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

# See some incorrectly-classes images
fig, axes = plt.subplots(nrows=3, ncols=3)
fig.tight_layout()
random.shuffle(incorrect_indices)
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
plt.show()