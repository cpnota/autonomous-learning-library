from setuptools import setup, find_packages

extras = {
    "docs": [
        "sphinx>=3.2.1",
        "sphinx-autobuild>=2020.9.1",
        "sphinx-rtd-theme>=0.5.0",
        "sphinx-automodapi>=0.13"
    ],
    "dev": [
        "pylint>=2.6.0",             # code quality tool
        "torch-testing>=0.0.2",      # pytorch assertion library
        "gym[atari,box2d]>=0.17.2",  # common environments
        "pybullet>=3.0.6"            # continuous environments
    ]
    }

extras["all"] = list(set().union(extras["docs"], extras["dev"]))

setup(
    name="autonomous-learning-library",
    version="0.6.1",
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
        "gym>=0.17.2",             # common environments
        "numpy>=1.18.0",           # math library
        "matplotlib>=3.3.0",       # plotting library
        "opencv-python>4.4.0.42",  # used by atari wrappers
        "torch>=1.5.1",            # deep learning
        "torchvision>=0.6.1"       # additional utilities
        "tensorboardX>=2.3.0",     # tensorboard compatibility
    ],
    extras_require=extras,
)
