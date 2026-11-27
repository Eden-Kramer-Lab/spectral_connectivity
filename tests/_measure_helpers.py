"""Settings and groupings of the registered measures shared by the test modules."""

from collections.abc import Sequence
from typing import Any

from spectral_connectivity._measure_registry import _MEASURE_SPECS, _is_group_measure

#: The measures an exact spectrum (``is_exact_spectrum``) refuses.
REFUSED_FOR_EXACT_SPECTRUM = [
    name for name, spec in _MEASURE_SPECS.items() if not spec.defined_for_exact_spectrum
]


def measure_kwargs(group_labels: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """The settings each registered measure needs to run, by name: the
    ``group_labels`` for a group measure, and one pair for the subset Granger."""
    kwargs: dict[str, dict[str, Any]] = {
        name: {"group_labels": group_labels}
        for name in _MEASURE_SPECS
        if _is_group_measure(name)
    }
    kwargs["subset_pairwise_spectral_granger_prediction"] = {"pairs": [(0, 1)]}
    return kwargs
