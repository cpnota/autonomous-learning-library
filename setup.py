from setuptools import setup, find_packages


extras = {
    "atari": [
        "atari_py~=0.2.0",            # atari environments
        "Pillow~=7.1.2",              # rendering library
    ],
    "box2d": [
        "box2d-py~=2.3.5",            # box3d physics environments
    ],
    "pybullet": [
        "pybullet>=3.0.6",            # open-source robotics environments
    ],
    "ma-atari": [
        "PettingZoo[atari]>=1.9.0",   # Multiagent atari environments
        "supersuit>=2.4.0",           # Multiagent env wrappers
        "AutoROM>=0.1.19",            # Tool for downloading ROMs
    ],
    "test": [
        "flake8>=3.8",                # linter for pep8 compliance
        "autopep8>=1.5",              # automatically fixes some pep8 errors
        "torch-testing>=0.0.2",       # pytorch assertion library
    ],
    "docs": [
        "sphinx>=3.2.1",              # documentation library
        "sphinx-autobuild>=2020.9.1", # documentation live reload
        "sphinx-rtd-theme>=0.5.0",    # documentation theme
        "sphinx-automodapi>=0.13",    # autogenerate docs for modules
    ],
    "comet": [
        "comet-ml>=3.2.11",           # experiment tracking using Comet.ml
    ]
}

extras["all"] = extras["atari"]  + extras["box2d"] + extras["pybullet"] + extras["ma-atari"] + extras["comet"]
extras["dev"] = extras["all"] + extras["test"] + extras["docs"] + extras["comet"]

setup(
    name="autonomous-learning-library",
    version="0.7.1",
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
        "gym~=0.18.0",             # common environment interface
        "numpy>=1.18.0",           # math library
        "matplotlib>=3.3.0",       # plotting library
        "opencv-python~=3.4.0",      # used by atari wrappers
        "torch~=1.8.0",            # core deep learning library
        "tensorboard>=2.3.0",      # logging and visualization
        "tensorboardX>=2.1.0",     # tensorboard/pytorch compatibility
        "cloudpickle>=1.2.0",      # used to copy environments
    ],
    extras_require=extras
)
