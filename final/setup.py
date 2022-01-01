#nsml: dacon/nia-pytorch:1.0

from distutils.core import setup

setup(
    name='ladder_networks',
    version='1.0',
    install_requires=[
        'torch_optimizer',
        'torch == 1.10.0',
        'torchaudio == 0.10.0',
        'pytorch-metric-learning',
        'faiss-gpu',
        'soundfile',
        'scipy',
        'audiomentations',
    ]
)