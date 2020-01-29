from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

perplexity = 10
X = np.array([[np.cos(theta), np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2*np.pi, 100)] + [[0,0,0],[0,0,0.1],[0,0.1,0],[0.1,0,0]])

# Colour map
colourlist = [10*[colour] for colour in ['black', 'grey', 'brown', 'red', 'orange', 'yellow', 'green', 'aqua', 'darkblue', 'deeppink']]
colourlist = [item for sublist in colourlist for item in sublist] + 4*['black']

# Apply T-SNE, plot
plt.figure(figsize=(10, 2))
plt.title('T-SNE, tilted circle, other arguments')
for i in range(5):
    X_embedded = TSNE(n_components=2, perplexity=perplexity, n_iter = 1000, verbose = 2).fit_transform(X)
    ax = plt.subplot(1, 5, i + 1)
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], s=15, c=colourlist)
plt.show()

# Plot X
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=20, c=colourlist)
plt.title('Original data')