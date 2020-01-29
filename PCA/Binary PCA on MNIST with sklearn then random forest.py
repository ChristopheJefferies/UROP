from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Two digits to consider
digit0, digit1 = 1, 8

# Load and format data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Pick out two digits in both training and testing data
index = []
for i, image in enumerate(y_train):
    if y_train[i]==digit0 or y_train[i]==digit1:
        index.append(i)
x_train, y_train = [x_train[i] for i in index], [y_train[i] for i in index]

index = []
for i, image in enumerate(y_test):
    if y_test[i]==digit0 or y_test[i]==digit1:
        index.append(i)
x_test, y_test = [x_test[i] for i in index], [y_test[i] for i in index]

# Standardize data
scaler = StandardScaler() # instance of a standardizer
scaler.fit(x_train) # now ready to use
x_train = scaler.transform(x_train) # applying the standardizer
x_test = scaler.transform(x_test) # applying the standardizer

# Make an instance of the PCA model, and fit to data
pca = PCA(n_components=2) # instance of the model
pca.fit(x_train)
x_train = pca.transform(x_train) # applying PCA
x_test = pca.transform(x_test) # applying PCA
print('Explained variance in PCA features: ', pca.explained_variance_ratio_) # This seems remarkably low...

# Colour map
colour_dict = {digit0:'r', digit1:'b'}
colour_list = [colour_dict[i] for i in list(y_train)]

# Plot
plt.figure()
plt.scatter(x_train[:, 0], x_train[:, 1], c=colour_list)
plt.title('MNIST PCA: digits %s, %s' %(digit0, digit1))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# Implement a random forest using the two main eigenvectors in feature space
myforest = RandomForestClassifier(max_depth=2, random_state=0)
myforest.fit(x_train, y_train)
RandomForestClassifier(bootstrap=True, verbose=1)
print('Random forest feature importances: ', myforest.feature_importances_)

#test accuracy
correct = sum(myforest.predict(x_test) == y_test)
print('Random forest accuracy percentage: ', 100*correct/len(y_test))