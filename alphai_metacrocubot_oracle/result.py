import numpy as np
import pandas as pd

from alphai_delphi.oracle.abstract_oracle import AbstractPredictionResult


class FeatureSensitivity:

    def __init__(self, feature, perturbations):
        self._feature = feature
        sensitivities = np.abs(perturbations)
        self._per_symbol_sensitivity = sensitivities
        self._average_sensitivity = np.nanmean(sensitivities)

    @property
    def feature(self):
        return self._feature

    @property
    def average_sensitivity(self):
        return self._average_sensitivity

    @property
    def per_symbol_sensitivity(self):
        return self._per_symbol_sensitivity


class OraclePrediction(AbstractPredictionResult):
    def __init__(self, mean_vector, lower_bound, upper_bound, prediction_timestamp, target_timestamp):
        """ Container for the oracle predictions.

        :param mean_vector: Prediction values for various series at various times
        :type mean_vector: pd.Series
        :param lower_bound: Lower edge of the requested confidence interval
        :type lower_bound: pd.Series
        :param upper_bound: Upper edge of the requested confidence interval
        :type upper_bound: pd.Series
        :param prediction_timestamp: Timestamp when the prediction was made
        :type target_timestamp: datetime
        """
        self.target_timestamp = target_timestamp
        self.mean_vector = mean_vector
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.prediction_timestamp = prediction_timestamp
        self.covariance_matrix = pd.DataFrame()
        self._feature_sensitivity = {}

    def add_feature_sensitivity(self, feature_sensitivity):
        """
        Add feature sensitivity value
        :param FeatureSensitivity feature_sensitivity:
        :return:
        """
        self._feature_sensitivity[feature_sensitivity.feature] = feature_sensitivity

    @property
    def features_average_sensitivity(self):
        """
        returns the dict with features as key and value as a sensitivity
        :return dict:
        """
        return {feature_name: sensitivity.average_sensitivity for feature_name, sensitivity in
                self._feature_sensitivity.items()}

    @property
    def features_per_symbol_sensitivity(self):
        return {feature_name: sensitivity.per_symbol_sensitivity for feature_name, sensitivity in
                self._feature_sensitivity.items()}

    @property
    def custom_metrics(self):
        return {
            'per_symbol_sensitivity': self.features_per_symbol_sensitivity,
            'average_sensitivity': self.features_average_sensitivity
        }

    def __repr__(self):
        return "<Oracle prediction: {}>".format(self.__dict__)
