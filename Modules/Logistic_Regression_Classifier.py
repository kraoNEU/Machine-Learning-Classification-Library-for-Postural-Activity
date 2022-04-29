import numpy as np


class BinaryLogisticRegression:
    """
    This class provides library for machine learning classification
    using Binary Logistic Regression Model.
    """

    def __init__(self, learning_rate=0.001, iteration_count=1000):
        """
        This method initializes the classification model by declaring
        all its fields.
        param learning_rate: The learning rate of the model
        param iteration_count: The count of number of iterations to train the model
        """
        self.learning_rate = learning_rate
        self.iteration_count = iteration_count
        self.model_weights = None
        self.model_bias = None
        self.input_data = None
        self.output_data = None
        self.sample_count = None
        self.feature_count = None

    def train_model(self, input_data, output_data):
        """
        This method helps the model in training using the input and output
        data from the training dataset.
        param input_data: The input samples data
        param output_data: The output data corresponding to the input samples
        """
        self.input_data = input_data
        self.output_data = output_data
        self.sample_count, self.feature_count = input_data.shape
        self.model_weights = np.zeros(self.feature_count)
        self.model_bias = 0

        for each_iteration in range(self.iteration_count):
            vector = np.vectorize(float)
            x = vector(input_data)
            model_expression = np.dot(x, self.model_weights) + self.model_bias
            predicted_model_output = self.logistic_function(model_expression)

            self.model_weights -= self.learning_rate * self.weights_error_calculator(predicted_model_output)
            self.model_bias -= self.learning_rate * self.bias_error_calculator(predicted_model_output)

    def weights_error_calculator(self, predicted_model_output2):
        """
        This method calculates the error expression for the weights of the
        model being trained.
        param predicted_model_output2: The output predicted by the model being trained
        return: The error of the weights of the model being trained
        """
        vector = np.vectorize(float)
        x1 = vector(self.input_data)
        error_in_model_weights = (1 / self.sample_count) * np.dot(x1.T,
                                                                  (predicted_model_output2 - self.output_data))
        return error_in_model_weights

    def bias_error_calculator(self, predicted_model_output3):
        """
        This method calculates the error expression for the bias of the
        model being trained.
        param predicted_model_output3: The output predicted by the model being trained
        return: The error of the bias of the model being trained
        """
        error_in_model_bias = (1 / self.sample_count) * np.sum(predicted_model_output3 - self.output_data)
        return error_in_model_bias

    @staticmethod
    def logistic_function(this_expression):
        """
        This method calculates the logistic / sigmoid function for each expression
        of the model being trained.
        param this_expression: The model expression for each input sample data
        return: The logistic function output for the input expression
        """
        return (1 / (1 + np.exp(-this_expression)))

    def predictor_trainer(self, input_data3):
        """
        This method is used to calculate the predicted
        output of the model.
        param input_data3: the input test data
        return: the output of the trained model using test data
        """
        model_expression = np.dot(input_data3, self.model_weights) + self.model_bias
        predicted_model_output1 = self.logistic_function(model_expression)
        return predicted_model_output1

    def class_predictor(self, input_data2):
        """
        This method predicts the output class for the corresponding
        test input sample by using the trained model.
        param input_data2: The test input sample data
        return: The output class of the test input sample data
        """
        predicted_model_class = None
        predicted_model_output = self.predictor_trainer(input_data2)
        predicted_model_class = [1 if each_output_sample > 0.5 else 0
                                 for each_output_sample in predicted_model_output]
        return predicted_model_class

    @staticmethod
    def percent_accuracy_of_model(output_model_class, predicted_model_class):
        """
        This method calculates the percentage accuracy of the model by comparing
        the ideal output with the predicted output of the model.
        param predicted_model_class: The predicted output of the model
        param output_model_class: The ideal output as per the dataset
        return: The percentage accuracy of the model
        """
        percent_accuracy: int = np.sum(output_model_class == predicted_model_class) / len(output_model_class) * 100
        return percent_accuracy


class MultiClassLogisticRegression:
    """
    This class provides library for machine learning classification
    using Multiclass Logistic Regression Model.
    """

    def __init__(self):
        """
        This method initializes the class of multiclass
        Logistic Regression.
        """
        self.class_model_dict = {}

    def train_model(self, input_data, output_data):
        """
        This method helps the model in training using the input and output
        data from the training dataset.
        param input_data: The input samples data
        param output_data: The output data corresponding to the input samples
        """
        target_class_list = list(set(output_data))
        for each_class in target_class_list:
            new_output_data = []
            for each_output_sample in output_data:
                if each_class == each_output_sample:
                    new_output_data.append(1)
                else:
                    new_output_data.append(0)
            this_model = BinaryLogisticRegression(learning_rate=0.0001, iteration_count=1000)
            this_model.train_model(input_data, new_output_data)
            self.class_model_dict[each_class] = this_model

    def class_predictor(self, input_data_new):
        """
        This method predicts the output class for the corresponding
        test input sample by using the trained model.
        param input_data_new: The test input sample data
        return: The output class list of the test input sample data
        """
        all_classes = self.class_model_dict.keys()
        predicted_class_list = []

        for each_example in input_data_new:
            probability_dict = {}
            for each in all_classes:
                model = self.class_model_dict[each]
                n = tuple(model.predictor_trainer(input_data_new))
                probability_dict[n] = each
            predicted_class = probability_dict.get(max(probability_dict.keys()))
            predicted_class_list.append(predicted_class)

        return predicted_class_list

    def accuracy(self, outputClass, predictedClass):
        """
        This method calculates the percentage accuracy of the model by comparing
        the ideal output with the predicted output of the model.
        param predictedClass: The predicted output of the model
        param outputClass: The ideal output as per the dataset
        return: The percentage accuracy of the model
        """
        percent_accuracy: int = np.sum(outputClass == predictedClass) / len(outputClass) * 100
        return percent_accuracy