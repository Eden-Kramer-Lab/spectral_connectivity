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

### How to build the documenation

1. Change directory to `spectral_connectivity/docs`
2. Run `make html` to preview the docs.
3. A commit to the master branch will automatically build the docs on readthedocs.

### How to make a release

1. Bump the version number and tag the commit
2. Upload to pypi

```bash
git clean -xfd
python -m build
twine upload dist/*
```

3. Upload to conda. This requires anaconda-client and conda-build.

```bash
# Build conda package using recipe
conda build conda-recipe/ --output-folder ./conda-builds

# Upload to your personal conda channel
anaconda upload ./conda-builds/noarch/spectral_connectivity-*.tar.bz2

# Clean up build artifacts
rm -rf ./conda-builds
conda build purge
```

**For conda-forge submission (future enhancement):**

This package is not currently on conda-forge. To add it:

1. **Prepare for submission:**
   - Ensure package has stable releases and good maintenance
   - Recipe should be well-tested with the conda-recipe/ directory

2. **Submit to conda-forge:**
   - Fork https://github.com/conda-forge/staged-recipes
   - Copy `conda-recipe/meta.yaml` to `recipes/spectral_connectivity/meta.yaml`
   - Submit PR to conda-forge staged-recipes
   - Respond to reviewer feedback

3. **After acceptance:**
   - conda-forge creates automated feedstock
   - Maintainers get notifications for new releases
   - Users can install with: `conda install -c conda-forge spectral_connectivity`

4. Release on github.

## Authorship on manuscripts

Authorship on any manuscripts for the `spectral_connectivity` package will be granted based on substantive contributions to the design and implementation of the spectral_connectivity package. This is not soley determined by lines of code or number of commits contributed to the project, but these will be considered when making this decision. For example, a one letter correction in documentation will not be considered substantive for authorship (although typo correction is very much appreciated).
