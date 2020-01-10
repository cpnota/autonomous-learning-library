from setuptools import setup, find_packages

setup(
    name="all",
    version="0.3.4",
    description=("A reinforcement learning library in python"),
    packages=find_packages(),
    url="https://github.com/cpnota/autonomous-learning-library.git",
    author="Autonomous Learning Lab",
    install_requires=[
        "gym[atari,box2d]",    # atari environments
        "numpy",         # math library
        "matplotlib",    # plotting library
        "opencv-python", # used by atari wrappers
        "pybullet",      # continuous environments
        "pylint",        # code quality tool
        "tensorboardX",  # visualize results
        "torch-testing", # testing library for pytorch
        # these should be installed globally:
        # "tensorflow",  # needed for tensorboard
        # "torch",       # deep learning library
        # "torchvision", # install alongside pytorch
    ],
)
