# Contributing

Contributions of all kinds are welcome. In particular, pull requests (PRs) are appreciated. The maintainers of this repository will help walk you through any issues in the pull request discussion, so please feel free to open a pull request even if you are new to pull requests.

## Issues

The easiest contribution to make is to [file an issue](https://github.com/Eden-Kramer-Lab/spectral_connectivity/issues). Please perform a search of [existing issues](https://github.com/Eden-Kramer-Lab/spectral_connectivity/issues?q=is%3Aissue) and provide clear instructions for how to reproduce a problem. If you have resolved an issue yourself, please contribute it to this repository so others can benefit from your work.

Please note that we **cannot**, in general, answer questions about particular connectivity measures and their merits. The user should be responsible for understanding the statistics they are using. Canonical papers for each connectivity measure are listed in the docstring of each connectivity measure. Questions and issues regarding implementation of the connectivity measures are welcome.

## Code

Code contributions are always welcome, from simple bug fixes to new features. To contribute code:

1. Please [fork the project](https://github.com/Eden-Kramer-Lab/spectral_connectivity/fork) into your own repository and make changes there. Follow the Developer Installation instructions in the README to set up an environment with all the necessary software packages.
2. Run code quality tools on your changes:
   - Format with [black](https://github.com/psf/black): `black spectral_connectivity/ tests/`
   - Lint with [ruff](https://github.com/astral-sh/ruff): `ruff check spectral_connectivity/ tests/`
   - Type check with [mypy](https://mypy.readthedocs.io/): `mypy spectral_connectivity/`
3. Add tests for bugs/new features and make sure existing tests pass. Tests will run through GitHub Actions.
4. Add docstrings for each function in the [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html).
5. Add references if you are adding a connectivity measure.
6. Update [CHANGELOG.md](CHANGELOG.md) with your changes under the "Unreleased" section.
7. Submit a pull request.

If you are fixing a known issue, please add the issue number to the PR message.

If you are fixing a new issue, file an issue and then reference it in the PR.

### How to build the documenation

1. Change directory to `spectral_connectivity/docs`
2. Run `make html` to preview the docs.
3. A commit to the master branch will automatically build the docs on readthedocs.

### How to make a release

This project uses an automated release workflow. To create a new release:

1. **Update CHANGELOG.md**
   - Move changes from "Unreleased" section to a new version section
   - Use format: `## [X.Y.Z] - YYYY-MM-DD`
   - Commit the changelog update

2. **Create and push a version tag**
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

3. **Automated workflow** (`.github/workflows/release.yml`)
   The release workflow will automatically:
   - Run code quality checks (black, ruff, mypy)
   - Run tests on Python 3.10, 3.11, 3.12, and 3.13
   - Build source distribution and wheel
   - Test the built packages
   - Publish to PyPI (requires trusted publishing setup)
   - Create a GitHub release with notes extracted from CHANGELOG.md

4. **Manual PyPI upload** (if needed)
   If you need to publish manually:
   ```bash
   python -m build
   twine check dist/*
   twine upload dist/*
   ```

5. **Conda release** (requires anaconda and conda-build)
   ```bash
   conda build conda-recipe/ --output-folder ./conda-builds
   anaconda upload ./conda-builds/noarch/spectral_connectivity-*.tar.bz2
   ```

### Version Numbering

This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backwards compatible manner
- **PATCH** version for backwards compatible bug fixes

## Authorship on manuscripts

Authorship on any manuscripts for the `spectral_connectivity` package will be granted based on substantive contributions to the design and implementation of the spectral_connectivity package. This is not soley determined by lines of code or number of commits contributed to the project, but these will be considered when making this decision. For example, a one letter correction in documentation will not be considered substantive for authorship (although typo correction is very much appreciated).
