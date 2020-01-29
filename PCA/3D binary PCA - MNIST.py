from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #for 3d projection. Used despite warning
import numpy as np

digit0, digit1 = 0, 1

# Load and format data
(images, labels), _ = mnist.load_data()
digitindex = np.nonzero([labels[i]==digit0 or labels[i]==digit1 for i in range(len(labels))])
images = images[digitindex]
labels = labels[digitindex]
images = images.reshape(len(images), 28*28)
num_images = 3000 # must be <= len(images)
x_train, y_train = images[:num_images], labels[:num_images]

# Standardize data, apply PCA
scaler = StandardScaler() # Standardizer
scaler.fit(x_train) # Standardize data
x_train = scaler.transform(x_train) # Apply transform (?)
pca = PCA(n_components=3) # Make instance of the model
pca.fit(x_train) # Fit PCA model to data
x_train = pca.transform(x_train) # Apply dimensionality reduction

# Colour map (lazy for now)
#colour_dict = {0:'black', 1:'grey', 2:'brown', 3:'red', 4:'orange', 5:'yellow', 6:'green', 7:'aqua', 8:'darkblue', 9:'deeppink'}
colour_dict = {digit0: 'red', digit1: 'darkblue'}
colour_list = [colour_dict[i] for i in list(y_train)]

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], s=5, c=colour_list)
plt.grid()
plt.title('PCA: MNIST (%s images), digits 0, 1' %num_images)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()