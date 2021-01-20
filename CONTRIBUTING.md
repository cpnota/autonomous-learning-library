# Contributing

Contributions and suggestions are welcome!
If you are interested in contributing either bug fixes or new features, open an issue and we can talk about it!
New PRs will require:

1. New unit tests for any new or changed common module, and all unit tests should pass.
2. All code should follow a similar style to the rest of the repository and the linter should pass.
3. Documentation of new features.
4. Manual approval.


We use the [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html) model, meaning that all PRs should be opened against the `develop` branch!
To begin, you can run the following commands:

```
git clone https://github.com/cpnota/autonomous-learning-library.git
cd autonomous-learning-library
git checkout develop
pip install -e .[docs]
```

The unit tests may be run using:

```
make test
```

You can automatically format your code to match our code style using:

```
make format
```

Finally, you rebuild the documentation using:

```
cd docs
make clean && make html
```

Happy hacking!
