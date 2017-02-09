from sklearn.datasets import load_iris

iris = load_iris()

# Classifiers are basically functions, f(X) = y
X = iris.data
y = iris.target

# Partition the data into training and testing sets
from sklearn.cross_validation import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# to test accuracy of training method
from sklearn.metrics import accuracy_score

# Classifying using decision trees
from sklearn import tree

my_clf = tree.DecisionTreeClassifier()

my_clf.fit(X_train, y_train)

predictions = my_clf.predict(X_test)

print(accuracy_score(y_test, predictions))

# Classifying using K-Nearest Neighbours
# from sklearn.neighbors import KNeighborsClassifier
import random

from scipy.spatial import distance

def euc(a, b):
	return distance.euclidean(a, b)

class ScrappyKNN():
	
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def closest(self, row):
		best_dist = euc(row, self.X_train[0])
		best_index = 0

		for i in range(1, len(self.X_train)):
			dist = euc(row, self.X_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i

		return self.y_train[best_index]

	def predict(self, X_test):
		predictions = []
		for row in X_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions

k_clf = ScrappyKNN()
k_clf.fit(X_train, y_train)

k_predictions = k_clf.predict(X_test)

print(accuracy_score(y_test, k_predictions))