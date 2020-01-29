from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #for 3d projection. Used despite warning

# Load and format data
(images, labels), _ = mnist.load_data()
images = images.reshape(60000, 784)
num_images = 3000 # <= 50000
x_train, x_test, y_train, y_test = images[:num_images], images[50000:], labels[:num_images], labels[50000:]

# Standardize the data
scaler = StandardScaler()

# Fit on training data
scaler.fit(x_train)

# Apply transform to both the training and testing data
x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)

# Make an instance of the model
pca = PCA(n_components=3)

# Fit PCA on training data
pca.fit(x_train)

# Apply the transform (dimensionality reduction)
x_train = pca.transform(x_train)
#x_test = pca.transform(x_test)

# Colour map (lazy for now)
colour_dict = {0:'black', 1:'grey', 2:'brown', 3:'red', 4:'orange', 5:'yellow', 6:'green', 7:'aqua', 8:'darkblue', 9:'deeppink'}
colour_list = [colour_dict[i] for i in list(y_train)]

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], s=5, c=colour_list)
plt.grid()
plt.title('Principal component analysis: MNIST (%s images)' %num_images)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()