from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #for 3d projection. Used despite warning
import numpy as np

digit0, digit1 = 0, 3
n_images = 1000 # Of each class

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

# Make data
trainindex = n_images*[digit0] + n_images*[digit1] #indices
np.random.shuffle(trainindex)
x_train = np.array([makeimage(i) for i in trainindex]) #make images according to indices
y_train = np.array(trainindex)

# Standardize data, apply PCA
scaler = StandardScaler() # Standardizer
scaler.fit(x_train) # Standardize data
x_train = scaler.transform(x_train) # Apply transform (?)
pca = PCA(n_components=3) # Make instance of the model
pca.fit(x_train) # Fit PCA model to data
x_train = pca.transform(x_train) # Apply dimensionality reduction

# Colour map (lazy for now)
colour_dict = {digit0:'red', digit1:'darkblue'}
colour_list = [colour_dict[i] for i in list(y_train)]

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], s=5, c=colour_list)
plt.grid()
plt.title('Principal component analysis: own data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()