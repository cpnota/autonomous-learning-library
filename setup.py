from setuptools import setup, find_packages

setup(
    name="autonomous-learning-library",
    version="0.4.0",
    description=("A library for building reinforcement learning agents in Pytorch"),
    packages=find_packages(),
    url="https://github.com/cpnota/autonomous-learning-library.git",
    author="Chris Nota",
    author_email="cnota@cs.umass.edu",
    install_requires=[
        "gym[atari,box2d]",     # common environments
        "numpy",                # math library
        "matplotlib",           # plotting library
        "opencv-python",        # used by atari wrappers
        "pybullet",             # continuous environments
        "tensorboardX",         # tensorboard compatibility
    ],
    extras_require={
        "pytorch": [
            "torch",            # deep learning
            "torchvision",      # additional utilities
            "tensorboard"       # visualizations
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            "sphinx-automodapi"
        ],
        "dev": [
            "pylint",           # code quality tool
            "torch-testing"     # pytorch assertion library
        ]
    },
)
