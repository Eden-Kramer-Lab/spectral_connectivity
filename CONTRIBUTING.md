# Contributing

Contributions of all kinds are welcome. In particular, pull requests (PRs) are appreciated. The maintainers of this repository will help walk you through any issues in the pull request discussion, so please feel free to open a pull request even if you are new to pull requests.

## Issues

The easiest contribution to make is to [file an issue](https://github.com/Eden-Kramer-Lab/spectral_connectivity/issues). Please perform a search of [existing issues](https://github.com/Eden-Kramer-Lab/spectral_connectivity/issues?q=is%3Aissue) and provide clear instructions for how to reproduce a problem. If you have resolved an issue yourself, please contribute it to this repository so others can benefit from your work.

Please note that we **cannot**, in general, answer questions about particular connectivity measures and their merits. The user should be responsible for understanding the statistics they are using. Canonical papers for each connectivity measure are listed in the docstring of each connectivity measure. Questions and issues regarding implementation of the connectivity measures are welcome.

## Code

Code contributions are always welcome, from simple bug fixes to new features. To contribute code:

1. Please [fork the project](https://github.com/Eden-Kramer-Lab/spectral_connectivity/fork) into your own repository and make changes there. Follow the Developer Installation instructions in the README to set up an environment with all the necessary software packages.
2. Run [black](https://github.com/python/black) and [flake8](http://flake8.pycqa.org/en/latest/) on your code.
3. Add tests for bugs/new features and make sure existing tests pass. Tests will run through github actions.
4. Add docstrings for each function in the [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html).
5. Add references if you are adding a connectivity measure.
6. Submit a pull request.

If you are fixing a known issue, please add the issue number to the PR message.

If you are fixing a new issue, file an issue and then reference it in the PR.
