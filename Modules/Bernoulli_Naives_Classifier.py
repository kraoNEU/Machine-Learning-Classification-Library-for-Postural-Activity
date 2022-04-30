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

        self.features = list
        self.likelihoods = {}
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
    def pre_processing(df):

        """
        function: Static Method for Getting the Target Classes and Input Features
        returns: Dataframe of Input Variables and Series of Target Classes
        """
        X = df.drop([df.columns[-1]], axis=1)
        y = df[df.columns[-1]]

        return X, y

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

    def fit(self, X, y):
        """
        function: Tries to fit the Features with the outcome classes of the Training Dataset
        returns: class posterior calculation and likelihood of the classes
        """
        self.features = list(X.columns)
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape[0]
        self.num_feats = X.shape[1]

        # Iterates over the features of the dataset
        for feature in self.features:

            # Defining a likelihood dictionary for further use for getting the mean and variance based on the
            # Input feature
            self.likelihoods[feature] = {}

            # Adding the features to the likelihood and class priors dictionaries to add the mean and variance later
            for outcome in np.unique(self.y_train):
                self.likelihoods[feature].update({outcome: {}})
                self.class_priors.update({outcome: 0})

        # Call the Method for Posterior Calculation and Likelihood Calculation
        self._calc_class_prior()
        self._calc_likelihoods()

    def _calc_class_prior(self):

        """
        function: calculates the posterior probability
        returns: the list of posterior probability outcomes for the given input
        """

        for outcome in np.unique(self.y_train):
            outcome_count = sum(self.y_train == outcome)
            self.class_priors[outcome] = outcome_count / self.train_size

    def _calc_likelihoods(self):
        """
        function: calculates the maximum possible likelihood of the features
        returns: list of the likelihood based on the mean and variance
        """
        for feature in self.features:

            for outcome in np.unique(self.y_train):

                # Getting the likelihoods of input variable "Mean"
                self.likelihoods[feature][outcome]['mean'] = self.X_train[feature][
                    self.y_train[self.y_train == outcome].index.values.tolist()].mean()

                # Getting the likelihoods of input variable "Variance"
                self.likelihoods[feature][outcome]['variance'] = self.X_train[feature][
                    self.y_train[self.y_train == outcome].index.values.tolist()].var()

    def predict(self, X):

        """
        Function: Predicts Whether the Input Features are for a particular class (target Classes) using Likelihood.
        Returns: The Array of Results for the Likelihood of the Target Classes
        Citation: https://towardsdatascience.com/naive-bayes-explained-9d2b96f4a9c0
        Formula and Basic Understanding have been inspired from the citation given above.
        """

        results = []
        X = np.array(X)

        # Iterating thru the Input features of X
        for query in X:

            # Storing the Posterior Outcome
            probs_outcome = {}

            for outcome in np.unique(self.y_train):
                prior = self.class_priors[outcome]
                likelihood = 1

                # Getting the Mean and Variance for the particular feature feature_outcome
                for feat, feat_val in zip(self.features, query):
                    mean = self.likelihoods[feat][outcome]['mean']
                    var = self.likelihoods[feat][outcome]['variance']
                    likelihood *= (1 / math.sqrt(2 * math.pi * var)) * np.exp(-(feat_val - mean) ** 2 / (2 * var))

                posterior_numerator = (likelihood * prior)
                probs_outcome[outcome] = posterior_numerator

            # Getting the final result of the Probability Outcome
            result = max(probs_outcome, key=lambda x: probs_outcome[x])

            # Append the results in the list
            # Appending the results into the List
            results.append(result)

        return np.array(results)


class MultiClassBernoulliClassification:

    def __init__(self):
        self.class_model_dict = {}

    def train_model(self, input_data, output_data):
        target_class_list = list(set(output_data))
        for each_class in target_class_list:
            new_output_data = []
            for each_output_sample in output_data:
                if each_class == each_output_sample:
                    new_output_data.append(1)
                else:
                    new_output_data.append(0)
            this_model = GaussianNB()
            this_model.fit(input_data, new_output_data)
            self.class_model_dict[each_class] = this_model

    def class_predictor(self, input_data_new):
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
        percent_accuracy: int = np.sum(outputClass == predictedClass) / len(outputClass) * 100
        return percent_accuracy