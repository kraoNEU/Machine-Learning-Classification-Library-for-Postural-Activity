import math
import numpy as np


class GaussianNB:
    """
    Function: Calculates the Gaussian Naives Bayes Classifier for the Input Dataset
    Static Methods: 3 Static Methods for Pre-Processing, Splitting and Accuracy Scoring
    Non-Static Methods: 4 Non-Static Method for fit, posterior calculation, likelihood estimations and predictions
    Input: Any Binary dataset with independent dataset (preferably)
    """

    def __init__(self):
        """Initializing the Variables"""

        self.dataSet_features = list
        self.likelihoods_lst = {}
        self.class_priors = {}

        self.X_train = np.array
        self.y_train = np.array
        self.train_size = int
        self.num_feats = int

    @staticmethod
    def accuracy_score(y_true, y_pred):
        """
        function: Static Method for the Calculation of Accuracy for the predicted output
        returns: Rounded Float Value of Accuracy Result
        """
        return round(float(sum(y_pred == y_true)) / float(len(y_true)) * 100, 2)

    @staticmethod
    def pre_processing(dataFrame):

        """
        function: Static Method for Getting the Target Classes and Input Features
        returns: Dataframe of Input Variables and Series of Target Classes
        """
        X_Data = dataFrame.drop([dataFrame.columns[-1]], axis=1)
        y_data = dataFrame[dataFrame.columns[-1]]

        return X_Data, y_data

    @staticmethod
    def train_test_split(x, y, test_train_split=0.25, random_state=None):
        """
         function: Static Method for the Splitting of Test and Train for the X and y Variables
         returns: x_train: Input features for training; x_test: Input features for testing
         y_train; Target class for training; y_test: Target class for Testing
        """
        x_test = x.sample(frac=test_train_split, random_state=random_state)
        y_test = y[x_test.index]

        x_train = x.drop(x_test.index)
        y_train = y.drop(y_test.index)

        return x_train, x_test, y_train, y_test

    def fit(self, X_Data, y_target_Data):
        """
        function: Tries to fit the Features with the outcome classes of the Training Dataset
        returns: class posterior calculation and likelihood of the classes
        """
        self.dataSet_features = list(X_Data.columns)
        self.X_train = X_Data
        self.y_train = y_target_Data
        self.train_size = X_Data.shape[0]
        self.num_feats = X_Data.shape[1]

        # Iterates over the features of the dataset
        for feature in self.dataSet_features:

            # Defining a likelihood dictionary for further use for getting the mean and variance based on the
            # Input feature
            self.likelihoods_lst[feature] = {}

            # Adding the features to the likelihood and class priors dictionaries to add the mean and variance later
            for outcome in np.unique(self.y_train):
                self.likelihoods_lst[feature].update({outcome: {}})
                self.class_priors.update({outcome: 0})

        # Call the Method for Posterior Calculation and Likelihood Calculation
        self.calculation_Posterior()
        self.calculation_Likelihood()

    def calculation_Posterior(self):

        """
        function: calculates the posterior probability
        returns: the list of posterior probability outcomes for the given input
        """

        for outcome in np.unique(self.y_train):
            outcome_count = sum(self.y_train == outcome)
            self.class_priors[outcome] = outcome_count / self.train_size

    def calculation_Likelihood(self):
        """
        function: calculates the maximum possible likelihood of the features
        returns: list of the likelihood based on the mean and variance
        """
        for feature in self.dataSet_features:

            for outcome in np.unique(self.y_train):
                # Getting the likelihoods of input variable "Mean"
                self.likelihoods_lst[feature][outcome]['mean'] = self.X_train[feature][
                    self.y_train[self.y_train == outcome].index.values.tolist()].mean()

                # Getting the likelihoods of input variable "Variance"
                self.likelihoods_lst[feature][outcome]['variance'] = self.X_train[feature][
                    self.y_train[self.y_train == outcome].index.values.tolist()].var()

    def predict(self, input_Dataset):

        """
        Function: Predicts Whether the Input Features are for a particular class (target Classes) using Likelihood.
        Returns: The Array of Results for the Likelihood of the Target Classes
        Citation: https://towardsdatascience.com/naive-bayes-explained-9d2b96f4a9c0
        Formula and Basic Understanding have been inspired from the citation given above.
        """

        results = []
        input_Dataset = np.array(input_Dataset)

        # Iterating thru the Input features of X
        for input_Data_Features in input_Dataset:

            # Storing the Posterior Outcome
            probability_Outcome_Dict = {}

            # Iterating thru the y_train
            for feature_outcomes in np.unique(self.y_train):
                prior = self.class_priors[feature_outcomes]

                # Setting the Input Feature Likelihood as 1 to change later
                feature_Likelihood = 1

                # Getting the Mean and Variance for the particular feature feature_outcome
                for dataset_features, dataset_feature_values in zip(self.dataSet_features, input_Data_Features):
                    # Calculating the Mean of the Input Features
                    calculate_Mean_feature = self.likelihoods_lst[dataset_features][feature_outcomes]['mean']

                    # Calculating the Variance of the Input Features
                    calculate_variance_feature = self.likelihoods_lst[dataset_features][feature_outcomes]['variance']

                    # Getting the Feature Input Likelihood
                    feature_Likelihood *= (1 / math.sqrt(2 * math.pi * calculate_variance_feature)) \
                                          * np.exp(-(dataset_feature_values - calculate_Mean_feature) ** 2 /
                                                   (2 * calculate_variance_feature))

                # Calculating the Posterior Numerator
                posterior_numerator = (feature_Likelihood * prior)

                # Storing the Dict value of the Mean and the Variance of the Input Feature
                probability_Outcome_Dict[feature_outcomes] = posterior_numerator

            # Getting the final result of the Probability Outcome
            result = max(probability_Outcome_Dict, key=lambda x: probability_Outcome_Dict[x])

            # Append the results in the list
            # Appending the results into the List
            results.append(result)

        # Returns the np.array of the results
        return np.array(results)


class MultiClassBernoulliClassification:
    """
    Function: Calculates the Multi-Gaussian Naives Bayes Classifier for the Input Dataset
    Static Methods: 2 Methods for train_model, class_predictor
    Input: Any Binary dataset with independent dataset (preferably)
    """

    def __init__(self):
        self.class_model_dict = {}

    def train_model(self, input_data_set, output_Prediction_Data):

        """
        function: calculates the one Vs rest methodology for Multi-Classification
        returns: list of the likelihood based on the mean and variance
        """

        target_feature_class = list(set(output_Prediction_Data))

        # Iterating thru the Target Feature Class
        for each_class in target_feature_class:

            new_output_class_List = []

            # Iterating thru the Prediction Data
            for each_output_sample in output_Prediction_Data:

                # Marking the Target Class as 1 and Rest as 0 in else
                if each_class == each_output_sample:
                    new_output_class_List.append(1)

                # Marking the other classes as 0
                else:
                    new_output_class_List.append(0)

            # Calling the GaussianNB for the 1 and 0 class
            this_model = GaussianNB()

            # Inputting the Dataset
            this_model.fit(input_data_set, new_output_class_List)

            # Storing the Model data in a list
            self.class_model_dict[each_class] = this_model

    def class_predictor(self, input_data_Class):

        """
        function: calculates the maximum possible likelihood of the features
        returns: list of the likelihood based on the mean and variance
        """

        # initialise the Dictionaries Keys
        all_classes = self.class_model_dict.keys()

        # Predicted Class List
        predicted_class_list = []

        # Iterating thru the Various Target Class
        for each_target_Class in input_data_Class:

            # Calculating the Posterior Probability for each
            probability_dict = {}

            # Iterating thru all the Target Classes
            for each in all_classes:

                # Getting the Values of each classes
                model = self.class_model_dict[each]

                # Getting the predictor of the each class
                n = tuple(model.predictor_trainer(input_data_Class))

                # Getting value of each of the values
                probability_dict[n] = each

            # getting the max value of the dictionry keys to get the probability
            predicted_class = probability_dict.get(max(probability_dict.keys()))

            # Gettingg all the predicted Class
            predicted_class_list.append(predicted_class)

        # Getting all the list of the generated class
        return predicted_class_list
