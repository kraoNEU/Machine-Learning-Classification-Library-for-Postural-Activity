from sklearn import datasets
from Modules import Logistic_Regression_Classifier as LRC
from sklearn.model_selection import train_test_split

test_dataset = datasets.load_breast_cancer()
test_input_data, test_output_data = test_dataset.data, test_dataset.target
input_train, input_test, output_train, output_test = train_test_split(test_input_data, test_output_data,
                                                                      test_size=0.2, random_state=1234)
test_model = LRC.BinaryLogisticRegression(learning_rate=0.0001, iteration_count=1000)
test_model.train_model(input_train, output_train)
test_prediction = test_model.class_predictor(input_test)
print("The percentage accuracy of the binary logistic regression model is: ",
      test_model.percent_accuracy_of_model(output_test, test_prediction))