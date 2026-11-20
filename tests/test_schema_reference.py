"""The cookbook's result-schema reference names every attribute a result carries.

The reference is read from ``docs/cookbook.md`` (the first column of its
attribute table, plus the ``mt_*`` compatibility keys it lists), so the table is
the only place the names are written down.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from spectral_connectivity import (
    MorletWavelet,
    Multitaper,
    ShortTimeFourierTransform,
    Welch,
    list_measures,
)
from spectral_connectivity.wrapper import (
    connectivity_to_xarray,
    fourier_connectivity,
    frequency_band_reduce,
    multitaper_connectivity,
)

_COOKBOOK = Path(__file__).parents[1] / "docs" / "cookbook.md"
_SECTION = "## Result schema and transform bandwidth"
_GROUP_CATEGORIES = {"group_pairwise", "multivariate_components"}
# Measures that need a uniform grid, which a Morlet transform does not have.
_UNIFORM_GRID_CATEGORIES = {"delay", "group_delay", "phase_slope"}


def _documented_patterns():
    """Regexes for the attribute names the schema reference documents."""
    text = _COOKBOOK.read_text(encoding="utf-8")
    section = text.split(_SECTION, 1)[1].split("\n## ", 1)[0]
    table_names = [
        name
        for row in section.splitlines()
        if row.startswith("| `")
        for name in re.findall(r"`([^`]+)`", row.split("|")[1])
    ]
    compatibility_keys = re.findall(r"`(mt_\w+)`", section)
    names = table_names + compatibility_keys
    # `arg_<name>` stands for any keyword name.
    return [re.compile(re.escape(name).replace("<name>", r"\w+") + "$") for name in names]


def _transforms(data):
    return {
        "multitaper": Multitaper(
            data, 64, time_halfbandwidth_product=2, time_window_duration=0.5
        ),
        "stft": ShortTimeFourierTransform(data, 64, time_window_duration=0.5),
        "welch": Welch(data, 64, segment_duration=0.5),
        "morlet": MorletWavelet(data, 64, frequencies=[4, 8, 16]),
    }


def _attribute_names(result):
    names = set(result.attrs)
    if isinstance(result, xr.Dataset):
        for variable in result.data_vars.values():
            names |= set(variable.attrs)
    return names


def _results(data):
    """Results of every transform and measure, and of each wrapper option."""
    labels = ["a", "a", "b", "b"]
    for transform_name, transform in _transforms(data).items():
        for info in list_measures():
            if transform_name == "morlet" and (
                info.requires_two_sided or info.category in _UNIFORM_GRID_CATEGORIES
            ):
                continue
            settings = {}
            if info.category in _GROUP_CATEGORIES:
                settings["group_labels"] = labels
            if info.name == "subset_pairwise_spectral_granger_prediction":
                settings["connectivity_kwargs"] = {"pairs": [(0, 1)]}
            yield connectivity_to_xarray(transform, info.name, **settings)

    common = {"sampling_frequency": 64, "time_window_duration": 0.5}
    yield multitaper_connectivity(
        data,
        **common,
        method=["power", "magnitude_squared_coherence", "delay"],
        frequency_range=(4, 20),
        frequency_decimation=2,
    )
    for reduction in ("mean", "integral"):
        yield multitaper_connectivity(
            data,
            **common,
            method="power",
            frequency_bands={"theta": (4, 8)},
            frequency_reduction=reduction,
        )
    with pytest.warns(DeprecationWarning, match="coherence_magnitude"):
        deprecated_name = multitaper_connectivity(data, **common, method="coherence_magnitude")
    yield deprecated_name
    multitaper = Multitaper(data, 64, time_window_duration=0.5)
    with pytest.warns(DeprecationWarning, match="frequencies_of_interest"):
        open_band = connectivity_to_xarray(
            multitaper, "delay", connectivity_kwargs={"frequencies_of_interest": (4, np.inf)}
        )
    yield open_band
    labeled = xr.DataArray(
        data,
        dims=("time", "trial", "signal"),
        coords={"time": np.arange(data.shape[0]) / 64, "subject": "s1"},
        attrs={"units": "uV", "note": "recorded"},
    )
    yield multitaper_connectivity(labeled, time_window_duration=0.5, method="power")
    yield multitaper_connectivity(
        data, **common, method="power", transform_kwargs={"tapers": np.ones((32, 1))}
    )
    coefficients = multitaper.fft()
    yield fourier_connectivity(coefficients, method="power")
    yield fourier_connectivity(coefficients, multitaper.frequencies, multitaper.time, "power")
    yield frequency_band_reduce(
        multitaper_connectivity(data, **common, method="coherency"), {"theta": (4, 8)}
    )


def test_schema_reference_names_every_result_attribute():
    """A result attribute the cookbook's schema reference does not name fails here:
    add it to the table in ``docs/cookbook.md``."""
    data = np.random.default_rng(0).standard_normal((128, 3, 4))
    patterns = _documented_patterns()
    with warnings.catch_warnings():
        # Warnings about the random data (orientation reminder, correlated
        # innovations, few observations) are not what this test is about.
        warnings.simplefilter("ignore", UserWarning)
        names = set().union(*(_attribute_names(result) for result in _results(data)))
    undocumented = sorted(
        name for name in names if not any(pattern.match(name) for pattern in patterns)
    )
    assert undocumented == []
    # The probe reaches the attributes that only some options write.
    for name in ("input_attrs_json", "frequency_bands_json", "frequency_coordinate"):
        assert name in names
