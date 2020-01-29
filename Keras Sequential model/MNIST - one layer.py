# Best run: 0.9824

#   Main influence was using a large layer for many epochs
#   Other parameter choices affected the accuracy very little

#   All 60000 images, 10 epochs
#   relu activation
#   Very large single layer
#   No regularization (more fiddling?)
#   Crossentropy loss, adam optimizer

# Imports
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils #for one-hot encoding

nb_classes = 10

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Turn 28x28 images in to 784-dim vector of floats. Scale inputs to 0-1 rather than 0-255
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Make labels have one-hot encoding
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Make the model
model = Sequential()
model.add(Dense(units=1000, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(units=10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train the model
model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_test, Y_test))

# Evaluate the model
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])