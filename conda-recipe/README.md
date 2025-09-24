# Conda Recipe for spectral_connectivity

This directory contains the conda recipe for building the `spectral_connectivity` package.

## Building the package

```bash
# Build the conda package
conda build conda-recipe/ --output-folder ./conda-builds

# Upload to anaconda.org (requires anaconda-client)
anaconda upload ./conda-builds/noarch/spectral_connectivity-*.tar.bz2
```

## Recipe Files

- `meta.yaml` - Main recipe file with package metadata, dependencies, and build configuration
- `build.sh` - Build script for Unix systems (Linux/macOS)

## Notes

- The recipe uses `noarch: python` for platform-independent builds
- Dependencies are synchronized with `pyproject.toml`
- Version is currently hardcoded but could be templated from git tags
- The recipe builds from PyPI source, not local source

## Conda-forge (Future Enhancement)

**Status: Not currently on conda-forge**

This package could benefit from conda-forge submission for:
- Automated builds across platforms
- Community maintenance
- Better discoverability
- Integration with conda-forge ecosystem

To submit to conda-forge:
1. Fork https://github.com/conda-forge/staged-recipes
2. Copy this `meta.yaml` to `recipes/spectral_connectivity/meta.yaml`
3. Submit pull request to staged-recipes
4. Address reviewer feedback
5. Once merged, conda-forge creates automated feedstock

**Benefits after conda-forge acceptance:**
- Users can install with: `conda install -c conda-forge spectral_connectivity`
- Automatic rebuilds for dependency updates
- Multi-platform builds (Windows, macOS, Linux)
- Community maintenance support