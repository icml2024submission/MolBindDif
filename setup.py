from setuptools import setup

setup(
    name="mbd",
    packages=[
        'data',
        'model',
        'experiments',
        'openfold',
        'linkpred'
    ],
    package_dir={
        'data': './data',
        'model': './model',
        'experiments': './experiments',
        'openfold': './openfold',
        'linkpred': './linkpred'
    },
)
