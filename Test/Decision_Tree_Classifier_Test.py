import math
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Reading the Dataset
my_data = pd.read_csv("../Dataset/posturalDatset.csv", delimiter=",")
my_data[0:5]

# Creating Input Vector X
X = my_data[['Tag', 'x', 'y', 'z']].values
X[0:4]

# Pre-Processing Tag Attribute
le_tag = preprocessing.LabelEncoder()
le_tag.fit(['010-000-024-033', '020-000-033-111', '020-000-032-221', '010-000-030-096'])
X[:, 0] = le_tag.transform(X[:, 0])

# y is the Goal Attribute
y = my_data["Activity"]
y[0:5]

# Train and Test Split for the Attribute Selection
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
postureTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
postureTree.fit(X_trainset, y_trainset)
predTree = postureTree.predict(X_testset)

# Print Test and Train
print(predTree[0:5])
print(y_testset[0:5])

accuracy_score = metrics.accuracy_score(y_testset, predTree)

# Evaluating the Accuracy
print("Decision Tree's Accuracy: ", accuracy_score)

# Testing the Decision
assert math.isclose(accuracy_score, 0.4896680011322738)