# Non-linear dimensionality reduction algorithm
# Aim: represent each high-dimensional data point with a low-dimensional point, so that 'similarity' is preserved
# Clustering is not to be trusted though; 'distance' not preserved

# Step 1: construct a probability distribution over PAIRS of data points, such that pairs of 'similar' points are likely
#       See formula online. sigma_i's are smartly chosen to make it sensible
# Step 2: construct a similar probability distribution over (arbitrary) pairs of points in a new, low-dimensional map
#       Then choose the location of these points by minimizing the Kulback-Leibler divergence ('relative entropy') of the second distribution from the first. (Use gradient descent)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

X = np.array([[0,0,0],[0,0,0.1],[0,0.1,0],[0.1,0,0], [0,0,1],[0,0,1.1],[0,0.1,1],[0.1,0,1], [0,1,1],[0,1,1.1],[0,1.1,1],[0.1,1,1], [1,1,1],[1,1,1.1],[1,1.1,1],[1.1,1,1]])
X_embedded = TSNE(n_components=2, perplexity=1).fit_transform(X)
print(X_embedded.shape)

# Colour map
colour_list = 4*['red'] + 4*['yellow'] + 4*['green'] + 4*['darkblue']

# Plot X
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=20, c=colour_list)
plt.title('Original data')

# Plot X_embedded
fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colour_list)
plt.title('T-SNE embedding')