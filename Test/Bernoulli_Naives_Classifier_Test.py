import math
import numpy as np
import pandas as pd
from Modules import Bernoulli_Naives_Classifier as BNC



# test_dataset = datasets.load_breast_cancer()
# test_input_data, test_output_data = test_dataset.data, test_dataset.target
# input_train, input_test, output_train, output_test = train_test_split(test_input_data, test_output_data,
#                                                                       test_size=0.2, random_state=1234)
# test_model = LRC.BinaryLogisticRegression(learning_rate=0.0001, iteration_count=1000)
# test_model.train_model(input_train, output_train)
# test_prediction = test_model.class_predictor(input_test)
# blr_accurcy = test_model.percent_accuracy_of_model(output_test, test_prediction)
# print("The percentage accuracy of the binary logistic regression model is: ",
#       blr_accurcy)
# assert math.isclose(blr_accurcy, 92.98245614035088)
#
# new_test_model = LRC.MultiClassLogisticRegression()
# new_test_model.train_model(input_train, output_train)
# new_test_prediction = new_test_model.class_predictor(input_test)
# mclr_accurcy = new_test_model.accuracy(output_test, new_test_prediction)
# print("The percentage accuracy of the multi class logistic regression model is: ",
#       mclr_accurcy)
# assert math.isclose(mclr_accurcy, 60.526315789473685)

data = pd.read_csv("/Users/cvkrishnarao/Desktop/Intro_Programming_Assignment/5010_Project/Dataset/Iris.csv")

# Split features and target
X, y = BNC.GaussianNB.pre_processing(data)

# Split data into Training and Testing Sets
X_train, X_test, y_train, y_test = BNC.GaussianNB.train_test_split(X, y, test_train_split=0.5)

# print(X_train, y_train)

gnb_clf = BNC.GaussianNB()
gnb_clf.fit(X_train, y_train)

# print(X_train, y_train)
accuracy_BNC = BNC.GaussianNB.accuracy_score(y_train, gnb_clf.predict(X_train))
print("Train Accuracy: {}".format(accuracy_BNC))

# Query 1:
query = np.array([[5.7, 2.9, 4.2, 1.3]])
print("Query 1:- {} ---> {}".format(query, gnb_clf.predict(query)))
assert gnb_clf.predict(query), ['Iris-setosa']