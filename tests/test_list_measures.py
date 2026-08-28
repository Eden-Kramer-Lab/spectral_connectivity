"""Discovery API for the connectivity measures the wrapper can compute."""

import pytest

from spectral_connectivity import Connectivity, list_measures
from spectral_connectivity.wrapper import (
    _MEASURE_SPECS,
    DEFAULT_METHODS,
    MeasureInfo,
)


def test_lists_every_registered_measure():
    """Every wrapper-supported measure is returned exactly once."""
    names = [measure.name for measure in list_measures()]
    assert names == list(_MEASURE_SPECS)


def test_each_measure_is_a_real_connectivity_method():
    """A listed name can always be called on the Connectivity class."""
    for measure in list_measures():
        assert hasattr(Connectivity, measure.name)


def test_returns_measureinfo_records_with_populated_fields():
    """Records expose name, category, description, and the two flags."""
    coherence = next(
        measure for measure in list_measures() if measure.name == "coherence_magnitude"
    )
    assert isinstance(coherence, MeasureInfo)
    assert coherence.category == "pairwise"
    assert coherence.is_default is True
    assert coherence.is_directed is False
    assert coherence.description == (
        "Return the magnitude squared of the complex coherency."
    )


def test_description_is_the_method_docstring_summary():
    """Descriptions are sourced from the method docstring, not duplicated."""
    for measure in list_measures():
        docstring = getattr(Connectivity, measure.name).__doc__ or ""
        first_line = docstring.strip().splitlines()[0].strip()
        assert measure.description == first_line
        assert measure.description  # never empty


def test_default_only_matches_default_methods():
    """The default-only view equals the exported DEFAULT_METHODS set."""
    names = [measure.name for measure in list_measures(default_only=True)]
    assert names == list(DEFAULT_METHODS)


def test_directed_filter_selects_directed_measures():
    """directed=True yields only directed measures; False only undirected."""
    directed = list_measures(directed=True)
    assert directed  # non-empty
    assert all(measure.is_directed for measure in directed)
    assert "pairwise_spectral_granger_prediction" in {
        measure.name for measure in directed
    }

    undirected = list_measures(directed=False)
    assert all(not measure.is_directed for measure in undirected)
    assert "coherence_magnitude" in {measure.name for measure in undirected}


def test_category_filter_selects_matching_output_kind():
    """category filters to measures with that output kind."""
    power = list_measures(category="power")
    assert [measure.name for measure in power] == ["power"]


def test_invalid_category_lists_valid_categories():
    """An unknown category raises and names the valid categories."""
    with pytest.raises(ValueError) as excinfo:
        list_measures(category="not_a_category")
    message = str(excinfo.value)
    assert "not_a_category" in message
    assert "pairwise" in message
