# Imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
#from keras.datasets import mnist



# Make the network object and add layers
#   Layer types: Dense, Dropout, ActivityRegularization...
#   Activations: sigmoid, tanh, relu, softmax...
model = Sequential()
model.add(Dense(units=64, activation='relu', activity_regularizer=regularizers.l2(0.01), input_shape=(100,))) # Only the first layer needs an input shape
model.add(Dropout(0.2))
model.add(Dense(units=10, activation='softmax'))



# Compile the model:
#   loss: can be a string identifier or an objective function
#       e.g. 'mse' for MSE regression problems, 'binary_crossentropy' or 'categorical_crossentropy' for those classification problems
#   optimizer: can be a string identifier or an instance of the Optimizer class
#       e.g. 'sgd', 'rmsprop' (good for recurrent networks), 'adam'
#   metrics: list of metrics to measure
#       e.g. 'accuracy'
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# For custom metrics
def mean_pred(y_true, y_pred):
    return keras.backend.mean(y_pred)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', mean_pred])



# Use training data in batches
#   x_train, y_train: Numpy arrays (if the model has a single input), or lists of Numpy arrays (multiple inputs)
#   batch_size: number of examples in each training batch
#   epochs: an epoch is one iteration over the whole x and y data
#   shuffle: boolean, 1 to shuffle the data before each epoch
model.fit(x_train, y_train, epochs=5, batch_size=32)



#Evaluate performance on test data. Outputs a list of scalars: [loss, metric1, metric2...]
loss_and_metrics = model.evaluate(x_test, y_test, epochs=10, batch_size=128)
print('Test loss:', loss_and_metrics[0])
print('Test accuracy:', loss_and_metrics[1])