# Imports
import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Load data
mnist = input_data.read_data_sets('MNIST_data')

# Make inputs for the discriminator (inputs_real) and generator (inputs_z)
def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, [None, real_dim], name='inputs_real')
    inputs_z = tf.placeholder(tf.float32, [None, z_dim], name='inputs_z')
    return inputs_real, inputs_z

# tf.variable_scope lets us reuse networks and outputs in different ways

# Build generator. z is the input tensor, n_units is the hidden layer size, alpha is the leaky ReLU parameter
def generator(z, out_dim, n_units=128, reuse=False,  alpha=0.01):

    with tf.variable_scope('generator', reuse = reuse):
        h1 = tf.layers.dense(z, n_units) # Hidden layer
        h1 = tf.maximum(alpha*h1, h1) # Leaky ReLU

        logits = tf.layers.dense(h1, out_dim)
        out = tf.tanh(logits) # tanh works best here
        return out, logits

# Build discriminator
def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse = reuse):
        h1 =tf.layers.dense(x, n_units) # Hidden layer
        h1 =tf.maximum(alpha*h1, h1) # Leaky ReLU

        logits =tf.layers.dense(h1, 1)
        out = tf.sigmoid(logits)
        return out, logits

# Hyperparameters
input_size = 784 # Size of input to discriminator
z_size = 100 # Size of latent vector to generator
g_hidden_size = 128
d_hidden_size = 128
alpha = 0.01 # Leaky ReLU parameter
smooth = 0.1 # Label smoothing

# Build network
tf.reset_default_graph()
input_real, input_z = model_inputs(input_size, z_size) # Input placeholders

g_model, g_logits = generator(input_z, out_dim=input_size) # g_model is the generator output. Reuse is false here...

# Make two discriminators, one for real data and one for fake data. Re-use weights variables between the two
d_model_real, d_logits_real = discriminator(input_real) # Real data discriminator
d_model_fake, d_logits_fake = discriminator(g_model, reuse=True) # Fake data discriminator

# Losses
# Fake discriminator loss using d_logits_fake. Take sigmoid cross-entropy, then mean over the whole batch. All labels zero.
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))
# Real discriminator loss using d_logits_real. All labels 0.9 (label smoothing to help generalize)
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels= tf.ones_like(d_logits_real)*(1-smooth)))
# Total discriminator loss
d_loss = d_loss_real + d_loss_fake
# Generator loss using d_logits_fake, all labels one. Will only use when passing a generated (fake) image in?
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

# Optimizers
# variable_scope made sure all our generator variable names start with 'generator', and the same for discriminator
learning_rate = 0.002
t_vars = tf.trainable_variables() # List of all trainable variables in our graph
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars) # minimize only updates the listed variables. So optimizers are separate
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

# Train
batch_size = 100
epochs = 100
samples = []
losses = []
saver = tf.train.Saver(var_list = g_vars)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for ii in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size) # Get next batch
            batch_images = batch[0].reshape((batch_size, 784)) # Reshape images
            batch_images = batch_images*2 - 1 # Scale from -1 to 1

            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size)) # Random noise for generator

            # Run optimizers
            _ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
            _ = sess.run(g_train_opt, feed_dict={input_z: batch_z})

        # At the end of each epoch, get the losses and print them out
        train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
        train_loss_g = g_loss.eval({input_z: batch_z})

        print("Epoch {}/{}...".format(e+1, epochs), "Discriminator Loss: {:.4f}...".format(train_loss_d), "Generator Loss: {:.4f}".format(train_loss_g))
        # Save losses to view after training
        losses.append((train_loss_d, train_loss_g))

        # Sample from generator as we're training for viewing afterwards
        sample_z = np.random.uniform(-1, 1, size=(16, z_size))
        gen_samples = sess.run(generator(input_z, input_size, reuse=True), feed_dict={input_z: sample_z})
        samples.append(gen_samples)
        saver.save(sess, './checkpoints/generator.ckpt')

# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)

# See training losses
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()

# See samples generated during training
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch][0]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    return fig, axes

# Load samples from generator taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)

# Samples from the final training epoch
_ = view_samples(-1, samples)

# Show images as network was training, every 10 epochs
rows, cols = 10, 6
fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes): #samples is a list of 100 tuples each of shape ((16, 784), (16, 784))
    for img, ax in zip(sample[::int(len(samples)/cols)], ax_row):
        ax.imshow(img[0].reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)