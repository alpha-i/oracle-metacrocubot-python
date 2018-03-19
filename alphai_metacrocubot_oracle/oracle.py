import logging

import numpy as np
from timeit import default_timer as timer

import alphai_crocubot_oracle.crocubot.evaluate as crocubot_eval
import pandas as pd
from alphai_crocubot_oracle.oracle import CrocubotOracle
from alphai_feature_generation.transformation import GymDataTransformation


class MetaCrocubotOracle(CrocubotOracle):

    def _init_data_transformation(self):
        data_trans_conf = self.config['data_transformation']
        data_trans_conf[GymDataTransformation.KEY_EXCHANGE] = self.calendar_name
        data_trans_conf["features_start_market_minute"] = self.scheduling['training_frequency']['minutes_offset']
        data_trans_conf["prediction_market_minute"] = self.scheduling['prediction_frequency']['minutes_offset']
        data_trans_conf["target_delta"] = self.prediction_horizon
        data_trans_conf["target_market_minute"] = self.scheduling['prediction_frequency']['minutes_offset']
        data_trans_conf["n_classification_bins"] = self.config['model']["n_classification_bins"]
        data_trans_conf["classify_per_series"] = self.config['model']["classify_per_series"]
        data_trans_conf["normalise_per_series"] = self.config['model']["normalise_per_series"]
        data_trans_conf["n_assets"] = self.config['model']["n_assets"]

        self._data_transformation = GymDataTransformation(data_trans_conf)

        self._target_feature = self._data_transformation.get_target_feature()
        self._n_features = len(self._data_transformation.features)

    def _sanity_check(self):
        pass

    def _init_universe_provider(self):
        pass

    def get_universe(self, data):
        return None

    def _preprocess_outputs(self, train_y_dict):

        train_y = list(train_y_dict.values())[0]
        train_y = np.swapaxes(train_y, axis1=1, axis2=2)

        if self._tensorflow_flags.predict_single_shares:
            n_feat_y = train_y.shape[3]
            train_y = np.reshape(train_y, [-1, 1, 1, n_feat_y])

        if self.network == 'dropout':
            train_y = np.squeeze(train_y)

        self.verify_y_data(train_y)

        return train_y.astype(np.float32)

    def _filter_universe_from_data_for_prediction(self, data, *args):
        return data

    def verify_pricing_data(self, predict_data):
        pass

    def predict(self, data, current_timestamp, target_timestamp):

        prediction_result = self._do_predict(data, current_timestamp, target_timestamp)

        # loop throught the feature, create a prediction to add sensitivity
        # for feature in feature_to_calculate
        #   feature_data = deepcopy(data)
        #   del feature_data[feature]
        #   result = self._do_predict(data, current_timestamp, target_timestamp)
        #   sensitivity = self._calculate_sensitivity(prediction_result, result)
        #   prediction_result.add_feature_sensitivity(feature, sensitivity)

        return prediction_result

    def _do_predict(self, data, current_timestamp, target_timestamp):
        """
             Main method that gives us a prediction after the training phase is done

             :param data: The dict of dataframes to be used for prediction
             :type data: dict
             :param current_timestamp: The timestamp of the time when the prediction is executed
             :type current_timestamp: datetime.datetime
             :param target_timestamp: The timestamp of the point in time we are predicting
             :type target_timestamp: datetime.datetime
             :return: Mean vector or covariance matrix together with the timestamp of the prediction
             :rtype: PredictionResult
             """

        if self._topology is None:
            logging.warning('Not ready for prediction - safer to run train first')

        logging.info('Crocubot Oracle prediction on {}.'.format(current_timestamp))

        self.verify_pricing_data(data)
        latest_train_file = self._train_file_manager.latest_train_filename(current_timestamp)
        predict_x, symbols, prediction_timestamp, target_timestamp = self._data_transformation.create_predict_data(data)

        logging.info('Predicting mean values.')
        start_time = timer()
        predict_x = self._preprocess_inputs(predict_x)

        if self._topology is None:
            n_timesteps = predict_x.shape[2]
            self.initialise_topology(n_timesteps)

        network_input_shape = self._topology.get_network_input_shape()
        data_input_shape = predict_x.shape[-3:]

        if data_input_shape != network_input_shape:
            err_msg = 'Data shape' + str(data_input_shape) + " doesnt match network input " + str(
                network_input_shape)
            raise ValueError(err_msg)

        predict_y = crocubot_eval.eval_neural_net(
            predict_x, self._topology,
            self._tensorflow_flags,
            latest_train_file
        )

        end_time = timer()
        eval_time = end_time - start_time
        logging.info("Network evaluation took: {} seconds".format(eval_time))

        means, conf_low, conf_high = self._data_transformation.inverse_transform_multi_predict_y(predict_y, symbols)
        self.log_validity_of_predictions(means, conf_low, conf_high)

        means_series = pd.Series(np.squeeze(means), index=symbols)
        conf_low_series = pd.Series(np.squeeze(conf_low), index=symbols)
        conf_high_series = pd.Series(np.squeeze(conf_high), index=symbols)

        return OraclePrediction(means_series, conf_low_series, conf_high_series, current_timestamp, target_timestamp)

    def log_validity_of_predictions(self, means, conf_low, conf_high):
        """ Checks that the network outputs are sensible. """

        if not (np.isfinite(conf_low).all() and np.isfinite(conf_high).all()):
            logging.warning('Confidence interval contains non-finite values.')

        if not np.isfinite(means).all():
            logging.warning('Means found to contain non-finite values.')

        logging.info('Samples from predicted means: {}'.format(means[0:10]))


class OraclePrediction:
    def __init__(self, mean_vector, lower_bound, upper_bound, prediction_timestamp, target_timestamp):
        """ Container for the oracle predictions.

        :param mean_vector: Prediction values for various series at various times
        :type mean_vector: pd.DataFrame
        :param lower_bound: Lower edge of the requested confidence interval
        :type lower_bound: pd.DataFrame
        :param upper_bound: Upper edge of the requested confidence interval
        :type upper_bound: pd.DataFrame
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

    def add_feature_sensitivity(self, feature, sensitivity):
        """
        Add feature sensitivity value
        :param feature:
        :param sensitivity:
        :return:
        """
        self._feature_sensitivity[feature] = sensitivity

    @property
    def features_sensitivity(self):
        return self._feature_sensitivity

    def __repr__(self):
        return "<Oracle prediction: {}>".format(self.__dict__)
