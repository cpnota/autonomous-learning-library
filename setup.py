from setuptools import find_packages, setup

GYMNASIUM_VERSION = "0.29.1"
PETTINGZOO_VERSION = "1.24.3"


extras = {
    "atari": [
        f"gymnasium[atari, accept-rom-license]~={GYMNASIUM_VERSION}",
    ],
    "pybullet": [
        "pybullet>=3.2.2,<4",
        "gym>=0.10.0,<0.26.0",
    ],
    "mujoco": [
        f"gymnasium[mujoco]~={GYMNASIUM_VERSION}",
    ],
    "ma-atari": [
        f"PettingZoo[atari, accept-rom-license]~={PETTINGZOO_VERSION}",
        "supersuit~=3.9.2",
    ],
    "test": [
        "black~=24.2.0",  # linting/formatting
        "isort~=5.13.2",  # sort imports
        "flake8~=7.0.0",  # more linting
        "torch-testing==0.0.2",  # pytorch assertion library
    ],
    "docs": [
        "sphinx~=7.2.6",  # documentation library
        "sphinx-autobuild~=2024.2.4",  # documentation live reload
        "sphinx-rtd-theme~=2.0.0",  # documentation theme
        "sphinx-automodapi~=0.17.0",  # autogenerate docs for modules
    ],
}

extras["all"] = (
    extras["atari"] + extras["mujoco"] + extras["pybullet"] + extras["ma-atari"]
)
extras["dev"] = extras["all"] + extras["test"]

setup(
    name="autonomous-learning-library",
    version="0.9.1",
    description=("A library for building reinforcement learning agents in Pytorch"),
    packages=find_packages(),
    url="https://github.com/cpnota/autonomous-learning-library.git",
    author="Chris Nota",
    author_email="cnota@cs.umass.edu",
    entry_points={
        "console_scripts": [
            "all-plot=all.scripts.plot:main",
            "all-atari=all.scripts.train_atari:main",
            "all-classic=all.scripts.train_classic:main",
            "all-continuous=all.scripts.train_continuous:main",
            "all-mujoco=all.scripts.train_mujoco:main",
            "all-multiagent-atari=all.scripts.train_multiagent_atari:main",
            "all-pybullet=all.scripts.train_pybullet:main",
            "all-watch-atari=all.scripts.watch_atari:main",
            "all-watch-classic=all.scripts.watch_classic:main",
            "all-watch-continuous=all.scripts.watch_continuous:main",
            "all-watch-mujoco=all.scripts.watch_mujoco:main",
            "all-watch-multiagent-atari=all.scripts.watch_multiagent_atari:main",
            "all-watch-pybullet=all.scripts.watch_pybullet:main",
        ],
    },
    install_requires=[
        f"gymnasium~={GYMNASIUM_VERSION}",  # common environment interface
        "numpy~=1.22",  # math library
        "matplotlib~=3.7",  # plotting library
        "opencv-python-headless~=4.0",  # used by atari wrappers
        "torch~=2.2",  # core deep learning library
        "tensorboard~=2.8",  # logging and visualization
        "cloudpickle~=2.0",  # used to copy environments
    ],
    extras_require=extras,
)
