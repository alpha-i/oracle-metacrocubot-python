from setuptools import setup
from setuptools import find_packages


setup(
    name='alphai_metacrocubot_oracle',
    version='0.0.1',
    description='Alpha-I Crocubot',
    author='Fergus Simpson',
    author_email='fergus.simpson@alpha-i.co',
    packages=find_packages(exclude=['doc', 'tests*']),
    install_requires=[
        'alphai_delphi>=2.0.0,<3.0.0',
        'alphai_crocubot_oracle==4.0.0'

    ],
    dependency_links=[
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_delphi/',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_crocubot_oracle/'
    ]
)
