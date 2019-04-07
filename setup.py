from setuptools import setup, find_packages


setup(
    name="all",
    version="0.0.2",
    description=("A reinforcement learning library in python"),
    packages=find_packages(),
    url="https://github.com/cpnota/autonomous-learning-library.git",
    author="Autonomous Learning Lab",
    install_requires=[
        "gym",
        "gym[atari]",
        "pylint",
        "tensorboardX",
        "torch-testing",
        # Install these yourself:
        # "numpy",      # everyone should have this
        # "tensorflow", # needed for tensorboard
        # "torch",      # neded to install carefully for CUDA
        # "torchvision" # should be installed alongisde torch
    ],
)
