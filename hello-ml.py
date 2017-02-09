from sklearn import tree

# Training data
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# Labels
labels = [0, 0, 1, 1]

# Get decision tree classifier
clf = tree.DecisionTreeClassifier()

# Train the model
clf.fit(features, labels)

# Skynet
print(clf.predict([[150, 1]]))
