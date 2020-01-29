# Imports
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import math
import random
import tensorflow as tf
from keras.utils import to_categorical

# Produces 8x8 image (as a list). Index 0 means colour in top left quadrant, 1 in top right, 2 in bottom left, 3 in bottom right
def makeimage(index):
    if index==0 or index==2:
        block = [[np.random.uniform(0.7, 1) for _ in range(4)] + 4*[0] for i in range(4)] #4x8, left side has colour
    else:
        block = [4*[0] + [np.random.uniform(0.7, 1) for _ in range(4)] for i in range(4)] #4x8, right side has colour
    if index==0 or index==1:
        image = block + [4*[0] for _ in range(8)] #8x8, top has noise
    else:
        image = [4*[0] for _ in range(8)] + block #8x8, bottom has noise
    return [item for mylist in image for item in mylist] #unpack list of lists

# Make train data
trainindex = [100*[i] for i in range(4)] #indices, 100 of each type
trainindex = [item for mylist in trainindex for item in mylist] #unpack list of lists
np.random.shuffle(trainindex)
trainimages = np.array([makeimage(i) for i in trainindex]) #make images according to indices
trainlabels = to_categorical(np.array(trainindex)) #one-hot encoding
# Make test data
testindex = [10*[i] for i in range(4)]
testindex = [item for mylist in testindex for item in mylist]
np.random.shuffle(testindex)
testimages = np.array([makeimage(i) for i in testindex])
testlabels = to_categorical(np.array(testindex))

# To run nicely in Jupyter notebook
sess = tf.InteractiveSession()

# Functions for creating weights, biases, convolutions and pooling
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID') #Change VALID to SAME to use padding. (Will mess up layer sizes below)

def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

# Create placeholder nodes for images and label inputs
x = tf.placeholder(tf.float32, shape=[None, 64])
y_ = tf.placeholder(tf.float32, shape=[None, 4])

# Input layer
x_image = tf.reshape(x, [-1,8,8,1])

# Conv layer 1: 10 3x3 filters
filtersize1 = 3
n_filters1 = 10
W_conv1 = weight_variable([filtersize1, filtersize1, 1, n_filters1])
b_conv1 = bias_variable([n_filters1])
x_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# Conv layer 2 - 5 2x2(x10) filters
filtersize2 = 3
n_filters2 = 7
W_conv2 = weight_variable([filtersize2, filtersize2, n_filters1, n_filters2])
b_conv2 = bias_variable([7])
x_conv2 = tf.nn.relu(conv2d(x_conv1, W_conv2) + b_conv2)

# Flatten (keras 'flatten')
finalsize = 8 - (filtersize1-1) - (filtersize2-1)
x_flat = tf.reshape(x_conv2, [-1, finalsize*finalsize*n_filters2])

# Dense fully connected layer
n_dense = 30
W_fc1 = weight_variable([finalsize*finalsize*n_filters2, n_dense])
b_fc1 = bias_variable([n_dense])
x_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

# Regularization with dropout
keep_prob = tf.placeholder(tf.float32)
x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)

# Classification layer and output
W_fc2 = weight_variable([n_dense, 4])
b_fc2 = bias_variable([4])
y_conv = tf.matmul(x_fc1, W_fc2) + b_fc2
y = tf.nn.softmax(y_conv)

# Loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Setup to test accuracy of model
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initilize all global variables
sess.run(tf.global_variables_initializer())

# Train model
n_epochs = 80
batch_size = 20 #dataset has 400 images
for epoch, _ in enumerate(range(n_epochs)):
    combined = list(zip(trainimages, trainlabels))
    random.shuffle(combined)
    trainimages[:], trainlabels[:] = zip(*combined)
    for i in range(math.floor(400/batch_size)):
        train_step.run(feed_dict={x: trainimages[10*i:10*i+batch_size], y_: trainlabels[10*i:10*i+batch_size], keep_prob: 0.4})
    train_accuracy = accuracy.eval(feed_dict={x: trainimages[10*i:10*i+40], y_: trainlabels[10*i:10*i+40], keep_prob: 1.0})
    print("Epoch %d, training accuracy %g"%(epoch, train_accuracy))

# Run trained model against test data
print("test accuracy %g"%accuracy.eval(feed_dict={x: testimages, y_: testlabels, keep_prob: 1.0}))

# Plot predictions
def plot_predictions(image_list, plot_number, output_probs=False, adversarial=False):
    prob = y.eval(feed_dict={x: image_list, keep_prob: 1.0}) #shape (40,4 in my case)
    predictions = np.zeros(len(image_list)).astype(int) #preallocation
    certainties = np.zeros(len(image_list)).astype(int) #preallocation

    # Set up image grid
    cols = 3
    rows = math.ceil(image_list.shape[0]/cols)
    fig = plt.figure(plot_number, (6., 6.))
    plt.clf()
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.5)

    # Get probs, images and populate grid
    for i in range(len(prob)):
        predictions[i] = np.argmax(prob[i])
        certainties[i] = prob[i][predictions[i]] * 100
        image = image_list[i].reshape(8,8)
        grid[i].imshow(image)
        grid[i].set_title('Label: {0} \nCertainty: {1}%' \
                          .format(predictions[i], certainties[i]))

        # Only use when plotting original, partial deriv and adversarial images
        if (adversarial) & (i % 3 == 1):
            grid[i].set_title("Adversarial \nPartial Derivatives")
            plt.title('Original image 0, adversarial target 3')
    plt.show()
    return prob if output_probs else None

# Plot some predictions
plot_predictions(np.array(testimages[:9]), plot_number=2)

# To generate adversarial images
def create_plot_adversarial_images(x_image, y_label, lr=0.1, n_steps=5, output_probs=False):
    original_image = x_image
    probs_per_step = []

    # Calculate loss, derivative and create adversarial image
    loss =  tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_conv)
    deriv = tf.gradients(loss, x)
    image_adv = tf.stop_gradient(x - tf.sign(deriv)*lr/n_steps)
    image_adv = tf.clip_by_value(image_adv, 0, 1) # prevents -ve values creating 'real' image

    for i in range(n_steps):
        # Calculate derivative and adversarial image
        dydx = sess.run(deriv, {x: x_image, keep_prob: 1.0}) # can't seem to access 'deriv' w/o running this
        x_adv = sess.run(image_adv, {x: x_image, keep_prob: 1.0})

        # Create darray of 3 images - orig, noise/delta, adversarial
        x_image = np.reshape(x_adv, (1, 64))
        img_adv_list = original_image # first image is original
        img_adv_list = np.append(img_adv_list, dydx[0], axis=0) # middle image is adversarial gradients
        img_adv_list = np.append(img_adv_list, x_image, axis=0) # last image is adversarial image

        # Print/plot images and return probabilities
        probs = plot_predictions(img_adv_list, output_probs=output_probs, adversarial=True, plot_number=i+3)
        probs_per_step.append(probs) if output_probs else None
    return probs_per_step

# Use a test image to generate an adversarial image
zeroindex = np.nonzero(testlabels[:][:,0])[0][0] #pick out a zero
chosenimage = testimages[zeroindex].reshape((1, 64))
chosenlabel = [0,0,0,1] # adversarial label 3

# Plot adversarial image
create_plot_adversarial_images(chosenimage, chosenlabel, lr=1, n_steps=8)

sess.close()