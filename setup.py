from setuptools import setup, find_packages


setup(
    name="all",
    version="0.0.1",
    description=("A reinforcement learning library in python"),
    packages=find_packages(),
    url="https://github.com/cpnota/autonomous-learning-library.git",
    author="Autonomous Learning Lab",
    install_requires=[
        "numpy",
        "gym",
        "pylint",
        "matplotlib"
    ],
)
