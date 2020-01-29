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

perplexitylist = [1, 5, 10, 40]
X = np.array([[np.cos(theta), np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2*np.pi, 100)])

# Colour map
colourlist = [10*[colour] for colour in ['black', 'grey', 'brown', 'red', 'orange', 'yellow', 'green', 'aqua', 'darkblue', 'deeppink']]
colourlist = [item for sublist in colourlist for item in sublist]

# Plot X
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=20, c=colourlist)
plt.title('Original data')

plt.figure()
plt.tight_layout(w_pad = 1.0, h_pad = 1.0)
for counter, perplexity in enumerate(perplexitylist):
    for i in range(3):
        X_embedded = TSNE(n_components=2, perplexity=perplexity, n_iter = 1000, verbose = 2).fit_transform(X)
        ax = plt.subplot(4, 3, 3*counter + i + 1)
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], s=2, c=colourlist)
        if i==0:
            plt.title('Perplexity %s' %perplexity)
plt.show()
