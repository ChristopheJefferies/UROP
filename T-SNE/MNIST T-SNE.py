# Non-linear dimensionality reduction algorithm
# Aim: represent each high-dimensional data point with a low-dimensional point, so that 'similarity' is preserved
# Clustering is not to be trusted though; 'distance' not preserved

# Step 1: construct a probability distribution over PAIRS of data points, such that pairs of 'similar' points are likely
#       See formula online. sigma_i's are smartly chosen to make it sensible
# Step 2: construct a similar probability distribution over (arbitrary) pairs of points in a new, low-dimensional map
#       Then choose the location of these points by minimizing the Kulback-Leibler divergence ('relative entropy') of the second distribution from the first. (Use gradient descent)

from keras.datasets import mnist
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#1000, 25, 800

n_points = 1000
perplexity = 30
n_iter = 2000

# Load and format data
(X_train, y_train), _ = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_train = X_train[:n_points]
X_embedded = TSNE(n_components=3, perplexity=perplexity, n_iter = n_iter, early_exaggeration = 10, verbose=2).fit_transform(X_train)

# Colour map
colour_dict = {0:'black', 1:'grey', 2:'brown', 3:'red', 4:'orange', 5:'yellow', 6:'green', 7:'aqua', 8:'darkblue', 9:'deeppink'}
colour_list = [colour_dict[i] for i in list(y_train)]

# Plot
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], s=4, c=colour_list[:n_points])
plt.grid()
plt.title('MNIST T-SNE: %s points, %s perplexity, %s iterations' %(n_points, perplexity, n_iter))