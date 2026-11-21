# Contributing

Contributions of all kinds are welcome. In particular, pull requests (PRs) are appreciated. The maintainers of this repository will help walk you through any issues in the pull request discussion, so please feel free to open a pull request even if you are new to pull requests.

## Issues

The easiest contribution to make is to [file an issue](https://github.com/Eden-Kramer-Lab/spectral_connectivity/issues). Please perform a search of [existing issues](https://github.com/Eden-Kramer-Lab/spectral_connectivity/issues?q=is%3Aissue) and provide clear instructions for how to reproduce a problem. If you have resolved an issue yourself, please contribute it to this repository so others can benefit from your work.

Please note that we **cannot**, in general, answer questions about particular connectivity measures and their merits. The user should be responsible for understanding the statistics they are using. Canonical papers for each connectivity measure are listed in the docstring of each connectivity measure. Questions and issues regarding implementation of the connectivity measures are welcome.

## Code

Code contributions are always welcome, from simple bug fixes to new features. To contribute code:

1. Please [fork the project](https://github.com/Eden-Kramer-Lab/spectral_connectivity/fork) into your own repository and make changes there. Follow the Developer Installation instructions in the README to set up an environment with all the necessary software packages.
2. Run code quality tools on your changes, through `uv run` so they are the
   versions pinned in `uv.lock` that CI uses:
   - Format with [ruff](https://github.com/astral-sh/ruff): `uv run ruff format src/ tests/`
   - Lint with [ruff](https://github.com/astral-sh/ruff): `uv run ruff check src/ tests/`
   - Type check with [mypy](https://mypy.readthedocs.io/): `uv run mypy src/`
3. Add tests for bugs/new features and make sure existing tests pass. Tests will run through GitHub Actions.
   Changes to backend-specific code must also pass the real-device smoke test on
   a CUDA machine: `SPECTRAL_CONNECTIVITY_ENABLE_GPU=true uv run --extra gpu
   pytest -m gpu`. Device-like mocks do not replace this release gate. The
   whole suite also runs on the GPU backend
   (`SPECTRAL_CONNECTIVITY_ENABLE_GPU=true uv run --extra gpu pytest`); keep it
   passing there. Tests that call private kernels pass device arrays with
   `to_device` and compare results after `to_host` (`tests/_backend_helpers.py`);
   tests of NumPy-only behavior take `@pytest.mark.cpu_only(reason=...)`, which
   is skipped on the GPU.
4. Add docstrings for each function in the [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html).
5. Add references if you are adding a connectivity measure. A new measure also
   needs:
   - a record in `src/spectral_connectivity/_measure_registry.py`, after which
     `python docs/generate_measure_table.py` regenerates the tables in
     `docs/CONNECTIVITY_METRIC_RANGES.md` (`tests/test_docs.py` fails while
     they are stale);
   - for a measure theoretically bounded in [0, 1], a clamp to that interval
     after computation, so rounding cannot leave values like 1 + 1e-12. Do
     not clamp unbiased estimators such as the pairwise phase consistency and
     the debiased phase-lag indices: their negative finite-sample values are
     legitimate;
   - for a phase-based measure, aggregation over the expectation (observation)
     axes only: trials, tapers, or segments.
6. Update [CHANGELOG.md](https://github.com/Eden-Kramer-Lab/spectral_connectivity/blob/master/CHANGELOG.md) with your changes under the "Unreleased" section.
7. Submit a pull request.

If you are fixing a known issue, please add the issue number to the PR message.

If you are fixing a new issue, file an issue and then reference it in the PR.

### How to build the documentation

1. Use Python >= 3.11: the docs need MyST-Parser >= 5.1, which does not
   install on 3.10. From the repository root, install the documentation
   dependencies with `pip install -r docs/requirements-docs.txt`.
2. Run `make -C docs html` to build the site. With uv,
   `uv run --with-requirements docs/requirements-docs.txt python -m sphinx -b html docs docs/_build/html`
   does steps 1 and 2 in one command.
3. Preview `docs/_build/html/index.html`. A commit to the master branch will
   automatically build the docs on Read the Docs.

### How to make a release

Releases are **automated**: pushing a `v*` tag runs the *Test, Build, and
Publish* workflow (`.github/workflows/release.yml`), which tests, builds the
sdist and wheel, generates build-provenance attestations, and publishes to PyPI
via PyPI **trusted publishing** (OIDC). No API token is stored in the
repository, and the default workflow token is read-only; only the publish and
release jobs elevate their permissions. Do **not** run `twine upload` by
hand — that bypasses the tests, attestations, and approval gate below.

To cut a release:

1. Update `CHANGELOG.md`: move the `[Unreleased]` entries under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading, and commit it.
2. If the release changes GPU code or dependencies, record a passing
   `SPECTRAL_CONNECTIVITY_ENABLE_GPU=true uv run --extra gpu pytest -m gpu` run
   from a CUDA machine in the release PR.
3. Create and push an annotated tag. The version is derived from the tag by
   `hatch-vcs`, so there is no version file to edit:

   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

4. The tagged run of the workflow:
   - runs the code quality checks (ruff format, ruff check, mypy);
   - runs the tests on Python 3.10, 3.11, 3.12, 3.13, and 3.14, plus the
     minimum dependency floors;
   - builds the source distribution and wheel and tests the built packages;
   - generates build-provenance attestations for the artifacts;
   - pauses at the protected `pypi` environment. An authorized maintainer
     approves it from the workflow run page; the artifacts then publish to PyPI
     (with attestations) and a GitHub Release is created from the changelog.

   If the automated publish fails, fix the workflow (or the PyPI
   trusted-publisher or environment settings) and re-run it; a manual upload
   races the workflow.

**One-time setup (required for the automated release to work):**

- On PyPI, configure a *trusted publisher* for this repository (PyPI project →
  Settings → Publishing), authorizing the OIDC publish without a stored token.
  Set the workflow to `release.yml` and the **environment name to `pypi`** to
  match the workflow's `environment: pypi`; the publish is rejected if they
  differ.
- In the GitHub repository (Settings → Environments → `pypi`), add protection
  rules — required reviewers, and optionally a tag/branch restriction. Naming
  the environment in the workflow is **not** sufficient on its own; the
  protection rules must be configured here so a publish needs manual approval.

Conda packages are published separately and manually. This requires
anaconda-client and conda-build.

Before building, update `conda-recipe/meta.yaml` for the new release, or it will
rebuild the previous version: set `version` to `X.Y.Z` and replace `sha256` with
the SHA-256 of the new PyPI sdist (from the release's "Download files" page, or
`openssl dgst -sha256 spectral_connectivity-X.Y.Z.tar.gz`), then commit it.

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

### Version Numbering

The version comes from the git tag through `hatch-vcs`: a tagged release is
`1.2.0`, and a development version is `1.2.0.dev5+g1a2b3c4` (5 commits since
the tag, plus the commit hash). This project follows
[Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backwards compatible manner
- **PATCH** version for backwards compatible bug fixes

## Authorship on manuscripts

Authorship on any manuscripts for the `spectral_connectivity` package will be granted based on substantive contributions to the design and implementation of the spectral_connectivity package. This is not solely determined by lines of code or number of commits contributed to the project, but these will be considered when making this decision. For example, a one letter correction in documentation will not be considered substantive for authorship (although typo correction is very much appreciated).
