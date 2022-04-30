import numpy as np
import pandas as pd
from Modules import Bernoulli_Naives_Classifier as BNC

data_binary = pd.read_csv("../Dataset/Iris.csv")
data_multi = pd.read_csv("../Dataset/posturalDatset.csv")

# Split features and target
X, y = BNC.GaussianNB.pre_processing(data_binary)

# Split data into Training and Testing Sets
X_train, X_test, y_train, y_test = BNC.GaussianNB.train_test_split(X, y, test_train_split=0.5)

gnb_clf = BNC.GaussianNB()
gnb_clf.fit(X_train, y_train)

# print(X_train, y_train)
accuracy_BNC = BNC.GaussianNB.accuracy_score(y_train, gnb_clf.predict(X_train))
print("Input Binary Classification Accuracy: {}".format(accuracy_BNC))

# Input Query for the Prediction Classifier
query = np.array([[5.7, 2.9, 4.2, 1.3]])
print("Input Binary Classification: {0} ---> {1}".format(query, gnb_clf.predict(query)))