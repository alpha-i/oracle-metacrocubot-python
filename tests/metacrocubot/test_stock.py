import tempfile
import unittest
import logging

import datetime

import os
import pytz

from alphai_delphi import Scheduler, OraclePerformance, Controller
from alphai_delphi.data_source.synthetic_data_source import SyntheticDataSource

from alphai_metacrocubot_oracle.datasource import DataSource
from alphai_metacrocubot_oracle.oracle import MetaCrocubotOracle

OUTPUT_DIR = tempfile.TemporaryDirectory().name
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), '..', 'resources')

logging.basicConfig(level=logging.DEBUG)


class TestMetaCrocubot(unittest.TestCase):

    def test_run(self):
        oracle_configuration = {
            'prediction_delta': {
                'unit': 'days',
                'value': 90
            },
            'training_delta': {
                'unit': 'days',
                'value': 180
            },
            'prediction_horizon': {
                'unit': 'days',
                'value': 30
            },
            'data_transformation': {
                'feature_config_list': [
                    {
                        'name': 'Returns',
                        'transformation': {
                            'name': 'value'
                        },
                        'normalization': 'standard',
                        'is_target': True,
                        'local': False,
                        'length': 5
                    },
                ],
                'features_ndays': 10,
                'features_resample_minutes': 15,
                'fill_limit': 5,
            },
            "model": {
                'train_path': OUTPUT_DIR,
                'covariance_method': 'NERCOME',
                'covariance_ndays': 9,
                'model_save_path': OUTPUT_DIR,
                'tensorboard_log_path': OUTPUT_DIR,
                'd_type': 'float32',
                'tf_type': 32,
                'random_seed': 0,

                # Training specific
                'predict_single_shares': True,
                'n_epochs': 1,
                'n_retrain_epochs': 1,
                'learning_rate': 2e-3,
                'batch_size': 100,
                'cost_type': 'bayes',
                'n_train_passes': 30,
                'n_eval_passes': 100,
                'resume_training': False,
                'classify_per_series': False,
                'normalise_per_series': False,

                # Topology
                'n_series': 3,
                'n_assets': 3,
                'n_features_per_series': 271,
                'n_forecasts': 1,
                'n_classification_bins': 12,
                'layer_heights': [270, 270],
                'layer_widths': [3, 3],
                'activation_functions': ['relu', 'relu'],

                # Initial conditions
                'INITIAL_ALPHA': 0.2,
                'INITIAL_WEIGHT_UNCERTAINTY': 0.4,
                'INITIAL_BIAS_UNCERTAINTY': 0.4,
                'INITIAL_WEIGHT_DISPLACEMENT': 0.1,
                'INITIAL_BIAS_DISPLACEMENT': 0.4,
                'USE_PERFECT_NOISE': True,

                # Priors
                'double_gaussian_weights_prior': False,
                'wide_prior_std': 1.2,
                'narrow_prior_std': 0.05,
                'spike_slab_weighting': 0.5,
            },
            "universe": {
                "method": "liquidity",
                "n_assets": 3,
                "ndays_window": 5,
                "update_frequency": 'weekly',
                "avg_function": 'median',
                "dropna": False
            },
        }

        scheduling_configuration = {
            "prediction_frequency": {"frequency_type": "WEEKLY", "days_offset": 0, "minutes_offset": 75},
            "training_frequency": {"frequency_type": "WEEKLY", "days_offset": 0, "minutes_offset": 60}
        }

        oracle = MetaCrocubotOracle(
            calendar_name="NYSE",
            oracle_configuration=oracle_configuration,
            scheduling_configuration=scheduling_configuration
        )

        simulation_start = datetime.datetime(2017, 5, 1, tzinfo=pytz.utc)
        simulation_end = datetime.datetime(2017, 9, 29, tzinfo=pytz.utc)
        calendar_name = 'NYSE'

        scheduler = Scheduler(
            simulation_start,
            simulation_end,
            calendar_name,
            oracle.prediction_frequency,
            oracle.training_frequency,
        )

        oracle_performance = OraclePerformance(OUTPUT_DIR, 'oracle')

        datasource = DataSource({
            'data_file': os.path.join(RESOURCES_DIR, 'test_stock_data')
        })

        controller = Controller(
            configuration={
                'start_date': '2017-05-01',
                'end_date': '2017-09-29'
            },
            oracle=oracle,
            scheduler=scheduler,
            datasource=datasource,
            performance=oracle_performance
        )

        controller.run()

        expected_files = ["oracle_correlation_coefficient.pdf",
                          "oracle_cumulative_returns.pdf",
                          "oracle_data_table.csv",
                          "oracle_oracle_results_actuals.hdf5",
                          "oracle_oracle_results_covariance_matrix.hdf5",
                          "oracle_oracle_results_mean_vector.hdf5",
                          "oracle_performance_table.csv",
                          "time-series-plot.pdf"
                          ]

        for filename in expected_files:
            self.assertTrue(os.path.isfile(os.path.join(OUTPUT_DIR, filename)))




