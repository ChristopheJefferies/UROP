# Imports
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Load data, ignoring labels (as autoencoders are self-supervised)
(x_train, _), (x_test, _) = mnist.load_data()

# Normalise and flatten the data
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Form layers (not yet in any Model)
input_img = Input(shape=(784,))
num_middle_neurons = 32
encoded = Dense(num_middle_neurons, activation='relu')(input_img) #32 gives compression factor of 24.5
decoded = Dense(784, activation='sigmoid')(encoded)

# Initialise the autoencoder using Model (functional API), so we can share layers with the encoder & decoder
autoencoder = Model(input_img, decoded) # Only state the first and last layers here

# Also create a separate encoder and decoder models
encoder = Model(input_img, encoded)

encoded_input = Input(shape=(num_middle_neurons,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Configure per-pixel crossentropy loss, then train the model
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=30, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Encode and decode some digits from the test set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Plot some inputs and outputs
plt.figure(figsize=(20, 4))
plt.title('MNIST autoencoder reconstruction: 32 middle neurons')
for i in range(10):
    # originals
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # reconstructions
    ax = plt.subplot(2, 10, i + 11)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()