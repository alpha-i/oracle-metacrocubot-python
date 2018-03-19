import os

import datetime

import pytz
from alphai_delphi import OraclePerformance, Controller, Scheduler

from alphai_metacrocubot_oracle.datasource import DataSource

CALENDAR_NAME = "JSEEOM"

OUTPUT_DIR = '/path/to/outputdir'

RESOURCES_DIR = os.path.join(os.path.dirname(__file__), 'tests', 'resources')

oracle_configuration = {
    'prediction_delta': {
        'unit': 'days',
        'value': 3
    },
    'training_delta': {
        'unit': 'days',
        'value': 12
    },
    'prediction_horizon': {
        'unit': 'days',
        'value': 1
    },
    'data_transformation': {
        'feature_config_list': [
            {
                'name': 'Returns',
                'transformation': {
                    'name': 'value'
                },
                'normalization': None,
                'is_target': True,
                'local': False,
                'length': 5
            },
        ],
        'features_ndays': 5,
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
        'n_series': 324,
        'n_assets': 324,
        'n_correlated_series': 1,
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
    }
}

scheduling_configuration = {
    "prediction_frequency": {"frequency_type": "MONTHLY", "days_offset": -1, "minutes_offset": 0},
    "training_frequency": {"frequency_type": "MONTHLY", "days_offset": -1, "minutes_offset": 0}
}

oracle = MetaCrocubotOracle(
    calendar_name=CALENDAR_NAME,
    oracle_configuration=oracle_configuration,
    scheduling_configuration=scheduling_configuration
)

simulation_start = datetime.datetime(2007, 12, 31, tzinfo=pytz.utc)
simulation_end = datetime.datetime(2010, 9, 29, tzinfo=pytz.utc)

scheduler = Scheduler(
    simulation_start,
    simulation_end,
    CALENDAR_NAME,
    oracle.prediction_frequency,
    oracle.training_frequency,
)

oracle_performance = OraclePerformance(OUTPUT_DIR, 'oracle')

datasource = DataSource({
    'data_file': os.path.join(RESOURCES_DIR, 'test_stock_data.hdf5')
})

controller = Controller(
    configuration={
        'start_date': simulation_start.strftime('%Y-%m-%d'),
        'end_date': simulation_end.strftime('%Y-%m-%d')
    },
    oracle=oracle,
    scheduler=scheduler,
    datasource=datasource,
    performance=oracle_performance
)

controller.run()
