# In scikit-learn, an estimator is any object that learns from data; could be for classification, regression, clustering...
# Classification task: put a given object in to one of finitely many classes. Regression: predict a continuous target variable
# The estimator itself will have some parameters we can choose.
# An estimator implements the methods fit(X, y), which fits the model, and predict(T), which outputs y for unlabeled T

# Linear regression: for continuous outcomes. Best match a linear model to given data using least squares. Does not give probabilities
# Logistic regression: for discrete outcomes. As above, then apply a link function so we can interpret the output as probabilities of different classes.
#   Hence this doesn't give too much weight to points far from the 'decision frontier' (?)

# Linear regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm

# Import and format data
iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]
n_sample = len(X)

# Shuffle data
np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order].astype(np.float)

X_train = X[:int(.9 * n_sample)]
y_train = y[:int(.9 * n_sample)]
X_test = X[int(.9 * n_sample):]
y_test = y[int(.9 * n_sample):]

# Fit model
clf = svm.SVC(kernel='linear', gamma=10)
clf.fit(X_train, y_train)

# Scatter plot
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired, edgecolor='k', s=20)
plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10, edgecolor='k') #circle out test data

# Colour plotting (copy-pasted)
plt.axis('tight')
x_min, x_max, y_min, y_max = X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()
XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
plt.title('Linear regression')
plt.show()