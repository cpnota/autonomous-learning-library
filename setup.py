from setuptools import setup, find_packages


def union(*deps):
    return list(set().union(*[extras[dep] for dep in deps]))

extras = {
    "envs": [
        "atari_py~=0.2.0",            # atari environments
        "box2d-py~=2.3.5",            # box2d environments
        "pybullet>=3.0.6",            # continuous environments
        "Pillow",                     # rendering library
    ],
    "test": [
        "pylint>=2.6.0",              # code quality tool
        "torch-testing>=0.0.2",       # pytorch assertion library
    ],
    "docs": [
        "sphinx>=3.2.1",              # documentation library
        "sphinx-autobuild>=2020.9.1", # documentation live reload
        "sphinx-rtd-theme>=0.5.0",    # documentation theme
        "sphinx-automodapi>=0.13",    # autogenerate docs for modules
    ]
}

extras["dev"] = union("envs", "test", "docs")


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
        "gym>=0.17.2",             # common environment interface
        "numpy>=1.18.0",           # math library
        "matplotlib>=3.3.0",       # plotting library
        "opencv-python>=3.,<4.",   # used by atari wrappers
        "torch>=1.5.1,<1.6",       # core deep learning library
        "tensorboard>=2.3.0",      # logging and visualization
        "tensorboardX>=2.1.0",     # tensorboard/pytorch compatibility
    ],
    extras_require=extras
)
