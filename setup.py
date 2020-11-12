from setuptools import setup, find_packages

setup(
    name="autonomous-learning-library",
    version="0.6.2",
    description=("A library for building reinforcement learning agents in Pytorch"),
    packages=find_packages(),
    url="https://github.com/cpnota/autonomous-learning-library.git",
    author="Chris Nota",
    author_email="cnota@cs.umass.edu",
    entry_points={
        'console_scripts': [
            'all-atari=scripts.atari:main',
            'all-classic=scripts.classic:main',
            'all-continuous=scripts.continuous:main',
            'all-plot=scripts.plot:main',
            'all-watch-atari=scripts.watch_atari:main',
            'all-watch-classic=scripts.watch_classic:main',
            'all-watch-continuous=scripts.watch_continuous:main',
            'all-benchmark-atari=benchmarks.atari40:main',
            'all-benchmark-pybullet=benchmarks.pybullet:main',
        ],
    },
    install_requires=[
        "gym[atari,box2d]",     # common environments
        "numpy",                # math library
        "matplotlib",           # plotting library
        "opencv-python>=3.,<4.",# used by atari wrappers
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
