import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Modules import Logistic_Regression_Classifier as LRC

test_dataset = datasets.load_breast_cancer()
test_input_data, test_output_data = test_dataset.data, test_dataset.target
input_train, input_test, output_train, output_test = train_test_split(test_input_data, test_output_data,
                                                                      test_size=0.2, random_state=1234)
test_model = LRC.BinaryLogisticRegression(learning_rate=0.0001, iteration_count=1000)
test_model.train_model(input_train, output_train)
test_prediction = test_model.class_predictor(input_test)
blr_accurcy = test_model.percent_accuracy_of_model(output_test, test_prediction)
print("The percentage accuracy of the binary logistic regression model is: ",
      blr_accurcy)
assert math.isclose(blr_accurcy, 92.98245614035088)

new_test_model = LRC.MultiClassLogisticRegression()
new_test_model.train_model(input_train, output_train)
new_test_prediction = new_test_model.class_predictor(input_test)
mclr_accurcy = new_test_model.accuracy(output_test, new_test_prediction)
print("The percentage accuracy of the multi class logistic regression model is: ",
      mclr_accurcy)


assert math.isclose(mclr_accurcy, 60.526315789473685)