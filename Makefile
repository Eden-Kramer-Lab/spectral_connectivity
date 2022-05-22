
make install:
    pip install conda
	conda update -q conda
    conda env create -f environment.yml
    source activate spectral_connectivity
	python install -e .

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
