import unittest

import os

from datetime import datetime, timedelta

import pandas as pd

from alphai_metacrocubot_oracle.datasource import DataSource

RESOURCES_DIR = os.path.join(os.path.dirname(__file__), 'resources')


class TestDataSource(unittest.TestCase):

    def setUp(self):

        configuration = {
            'data_file': os.path.join(RESOURCES_DIR, 'test_stock_data.hdf5')
        }

        self.datasource = DataSource(configuration)

    def test_data_source_get_data(self):

        data = self.datasource.get_data(datetime(2017, 9, 29), timedelta(days=30))

        expected_keys = set([
            'Shares',
            'Returns',
            'Beta to J203',
            'Beta to USDZAR',
            'Beta to RBAS',
            'Beta to RLRS',
            'Beta to FSPI',
            'Specific Risk',
            'Debt to Equity',
            'Debt to Equity Trend',
            'Interest Cover Trend',
            'log Market Cap',
            'log Trading Volume',
            'Earnings Yield',
            'Earnings Yield Trend',
            'Bookvalue to Price',
            'Bookvalue to Price Trend',
            'Dividend Yield',
            'Dividend Yield Trend',
            'Cashflow to Price',
            'Cashflow to Price Trend',
            'Sales to Price',
            'Sales to Price Trend',
            'Profit Margin',
            'Capital Turnover',
            'Capital Turnover Trend',
            'Return on Assets',
            'Return on Assets Trend',
            'Return on Equity',
            '3 Month Return',
            '6 Month Return',
            '12 Month Return',
            '24 Month Return',
            '36 Month Return',
            'Resource',
            'Financial'])

        assert set(data.keys()) == expected_keys

        returns = data['Returns']

        all_symbols = set([
            '~ABL',
            'ACL',
            'ACP',
            'ACT',
            'ADH',
            'ADI',
            'ADR',
            'ADW',
            '~AEA',
            'AEE',
            'AEG',
            'AEL',
            'AFE',
            'AFR',
            'AFT',
            'AFX',
            'AGI',
            'AGL',
            '~AIA',
            'AIP',
            'ALH',
            '~ALM',
            'ALT',
            'AMA',
            'AME',
            'AMS',
            'AND',
            'ANG',
            'ANS',
            'APF',
            '~APK',
            'APN',
            '~AQP',
            'ARH',
            'ARI',
            'ARL',
            'ART',
            'ASC',
            'ASR',
            'ATL',
            'ATT',
            'AVI',
            'AVU',
            'AVV',
            '~AWB',
            'BAT',
            'BAW',
            'BCF',
            'BCX',
            'BDM',
            'BEG',
            'BEL',
            'BFS',
            'BGA',
            'BIL',
            'BLU',
            'BRT',
            'BSR',
            'BSS',
            'BTI',
            'BVT',
            'CAT',
            'CBH',
            'CCO',
            '~CDZ',
            'CFR',
            'CGR',
            'CIL',
            'CKS',
            'CLH',
            'CLI',
            'CLR',
            'CLS',
            'CMH',
            'CML',
            'CMP',
            'CND',
            'CNL',
            'COH',
            'COM',
            'CPI',
            'CPL',
            'CRM',
            'CSB',
            'CSG',
            '~CSP',
            'CUL',
            'CVH',
            'CVI',
            'CZA',
            'DAW',
            '~DCT',
            '~~DGC',
            'DIA',
            'DLT',
            'DLV',
            '~DMC',
            'DRD',
            'DST',
            'DSY',
            'DTA',
            'DTC',
            'EHS',
            'ELI',
            'ELR',
            'EMH',
            'EMI',
            'ENX',
            'EOH',
            'EPS',
            '~ERB',
            'ESR',
            'EXG',
            'EXL',
            'EXX',
            'FBR',
            'FFA',
            'FGL',
            'FPT',
            '~FRT',
            'FSR',
            'FUU',
            'FVT',
            '~GBG',
            'GDO',
            'GFI',
            '~GGM',
            'GIJ',
            'GLN',
            'GMB',
            'GND',
            'GPA',
            'GPL',
            'GRF',
            'GRT',
            'GTR',
            'HAR',
            'HCI',
            'HDC',
            'HLM',
            '~HPA',
            'HSP',
            'HUG',
            'HWA',
            'HWN',
            'HYP',
            'IAP',
            '~ILA',
            '~ILV',
            'IMP',
            'ING',
            'INL',
            'INP',
            'IPF',
            'IPL',
            'ITE',
            'ITU',
            'IVT',
            'IWE',
            'JBL',
            'JDG',
            'JSC',
            'JSE',
            'KAP',
            '~KEH',
            'KEL',
            'KGM',
            'KIO',
            'LAF',
            'LBH',
            'LEW',
            'LHC',
            'LHG',
            'LON',
            '~MAS',
            '~~MDC',
            'MDI',
            'MFL',
            'MIX',
            'MMG',
            'MMH',
            'MMI',
            '~MML',
            'MMP',
            'MND',
            'MNP',
            'MOB',
            'MOR',
            'MPT',
            'MRF',
            'MRP',
            'MSM',
            'MSP',
            'MST',
            'MTA',
            'MTE',
            'MTL',
            'MTN',
            'MTX',
            'MUR',
            'MVL',
            'NBC',
            'NED',
            'NEP',
            'NHM',
            'NIV',
            'NPK',
            'NPN',
            'NT1',
            'NTC',
            'NWL',
            'OAO',
            'OAS',
            'OCE',
            'OCT',
            'OLG',
            'OML',
            'OMN',
            'PAM',
            'PAN',
            'PAP',
            'PBG',
            'PCN',
            '~PET',
            'PFG',
            'PGL',
            'PGR',
            'PHM',
            'PIK',
            'PMM',
            '~PNG',
            'PPC',
            'PPE',
            'PSG',
            '~PWK',
            'QHL',
            'RAH',
            'RBP',
            'RBX',
            'RCL',
            'RDF',
            'REB',
            'REI',
            'REM',
            'RES',
            'RLF',
            'RLO',
            'RMH',
            'RMI',
            'RNG',
            'ROC',
            'RPL',
            'RSG',
            '~SAB',
            'SAC',
            '~SAH',
            'SAL',
            'SAP',
            'SAR',
            'SBK',
            'SBL',
            'SCL',
            'SCP',
            'SDH',
            'SEP',
            'SFN',
            'SGL',
            'SHP',
            'SIM',
            'SLM',
            'SNH',
            'SNT',
            'SNV',
            'SOH',
            'SOL',
            'SOV',
            'SPG',
            'SPP',
            'SSK',
            'STP',
            'SUI',
            'SUR',
            '~SYC',
            'TAS',
            'TAW',
            'TBG',
            'TBS',
            'TCP',
            'TDH',
            'TEX',
            'TFG',
            'THA',
            'TKG',
            'TMT',
            'TON',
            'TOR',
            'TPC',
            'TRE',
            'TRU',
            'TSH',
            'TSX',
            'TTO',
            'TWR',
            'UCP',
            'UCS',
            'UNI',
            'UUU',
            'VIF',
            'VIL',
            'VKE',
            'VLE',
            'VOD',
            'VOX',
            'VUN',
            'WBO',
            'WEZ',
            'WGR',
            'WHL',
            'WIL',
            'WSL',
            'YRK',
            'ZCI',
            'ZED',
            'ZSA'])
        assert isinstance(returns, pd.DataFrame)

        assert set(returns.columns).issubset(all_symbols)


    def test_values_for_symbols_feature_and_time(self):

        values = self.datasource.values_for_symbols_feature_and_time(
         ['HCI', 'SNT'],
         'Returns',
         datetime(2017, 9, 29)
        )

        assert values['HCI'] == -0.292391161
        assert values['SNT'] == -0.137834485





