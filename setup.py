from setuptools import setup, find_packages


GYM_VERSION = "0.23.1"
PETTINGZOO_VERSION = "1.17.0"


extras = {
    "atari": [
        "gym[atari, accept-rom-license]~={}".format(GYM_VERSION),
    ],
    "box2d": [
        "gym[box2d]~={}".format(GYM_VERSION),
    ],
    "pybullet": [
        "pybullet>=3.2.2",
    ],
    "ma-atari": [
        "PettingZoo[atari]~={}".format(PETTINGZOO_VERSION),
        "supersuit~=3.3.5",
    ],
    "test": [
        "flake8>=3.8",                 # linter for pep8 compliance
        "autopep8>=1.5",               # automatically fixes some pep8 errors
        "torch-testing>=0.0.2",        # pytorch assertion library
    ],
    "docs": [
        "sphinx>=3.2.1",               # documentation library
        "sphinx-autobuild>=2020.9.1",  # documentation live reload
        "sphinx-rtd-theme>=0.5.0",     # documentation theme
        "sphinx-automodapi>=0.13",     # autogenerate docs for modules
    ],
    "comet": [
        "comet-ml>=3.28.3",            # experiment tracking using Comet.ml
    ]
}

extras["all"] = extras["atari"]  + extras["box2d"] + extras["pybullet"] + extras["ma-atari"] + extras["comet"]
extras["dev"] = extras["all"] + extras["test"] + extras["docs"] + extras["comet"]

setup(
    name="autonomous-learning-library",
    version="0.8.1",
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
        "gym~={}".format(GYM_VERSION),             # common environment interface
        "numpy>=1.22.3",           # math library
        "matplotlib>=3.5.1",       # plotting library
        "opencv-python~=3.4.0",    # used by atari wrappers
        "torch>=1.11.0",            # core deep learning library
        "tensorboard>=2.8.0",      # logging and visualization
        "cloudpickle>=2.0.0",      # used to copy environments
    ],
    extras_require=extras
)
