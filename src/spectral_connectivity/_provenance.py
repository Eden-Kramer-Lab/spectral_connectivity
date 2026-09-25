"""NetCDF-safe provenance attributes recorded on the wrapper's results."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.utils import get_compute_backend, to_numpy


def _json_compatible(value: Any) -> Any:
    """Convert provenance values to deterministic, JSON-compatible objects.

    JSON-native values convert deterministically. NumPy arrays/scalars and
    Python sequences are normalized to their JSON value representation, so
    container and dtype distinctions are not retained. Mappings with non-string
    keys use a tagged item-list representation so unlike keys such as ``1`` and
    ``"1"`` cannot collide. Any other object is recorded on a best-effort basis
    as ``{"python_type", "repr"}``; that ``repr`` is not guaranteed stable
    across runs because it may embed a memory address.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return {"nonfinite_float": repr(value)}
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, np.ndarray):
        # Host-convert via the util so a device-backed array is never moved by
        # an implicit ``np.asarray``/``tolist`` transfer.
        return _json_compatible(to_numpy(value).tolist())
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _json_compatible(item) for key, item in sorted(value.items())}
        converted_items = [
            [_json_compatible(key), _json_compatible(item)] for key, item in value.items()
        ]
        # Keys are already JSON-compatible; sort by their canonical form so the
        # serialization contract lives in one place (``_canonical_json``).
        converted_items.sort(key=lambda pair: _canonical_json(pair[0]))
        return {"python_type": "mapping", "items": converted_items}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_compatible(item) for item in value]
    return {
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _canonical_json(value: Any) -> str:
    """Serialize provenance as stable JSON suitable for a NetCDF attribute."""
    return json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _netcdf_provenance_value(value: Any) -> Any:
    """Return a NetCDF-safe scalar, using JSON for structured values.

    Non-finite floats are encoded as JSON too, so the ``arg_<key>`` view matches
    the ``measure_kwargs_json`` record rather than storing a bare ``NaN``/``inf``
    that not every NetCDF engine round-trips cleanly. Booleans become 0/1
    integers: netCDF4 and h5netcdf reject boolean attributes, and NetCDF3
    silently turns them into int8.
    """
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return _canonical_json(value)
    if isinstance(value, (str, int, float, np.integer, np.floating)):
        return value
    return _canonical_json(value)


def _store_provenance_item(attrs: dict[str, Any], prefix: str, key: Any, value: Any) -> None:
    """Record ``value`` under ``<prefix><key>`` as a NetCDF-safe attribute.

    A scalar is stored as-is; a structured or non-finite value is stored as a
    canonical JSON string under ``<prefix><key>_json`` so a consumer knows to
    ``json.loads`` it (mirrors the ``arg_<key>`` / ``arg_<key>_json`` split).
    Raises ``ValueError`` on a name collision (e.g. a structured ``x`` and a
    scalar ``x_json`` both mapping to ``<prefix>x_json``) rather than silently
    overwriting the earlier value; ``measure_kwargs_json`` remains the canonical
    full record.
    """
    netcdf_value = _netcdf_provenance_value(value)
    if isinstance(netcdf_value, str) and not isinstance(value, str):
        attr_name = f"{prefix}{key}_json"
    else:
        attr_name = f"{prefix}{key}"
    if attr_name in attrs:
        msg = (
            f"Provenance attribute {attr_name!r} is assigned twice; two keys "
            f"(one of them {key!r}) collide under the {prefix!r} namespace. "
            "Rename the offending argument."
        )
        raise ValueError(msg)
    attrs[attr_name] = netcdf_value


def _package_version() -> str:
    """Return the installed spectral_connectivity version (or ``"unknown"``)."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("spectral_connectivity")
    except (PackageNotFoundError, ImportError):
        # Package not installed / metadata unavailable; a genuinely unexpected
        # error is left to surface rather than being masked as "unknown".
        return "unknown"


def _shared_provenance_attrs(
    connectivity: Connectivity,
    transform_metadata: Mapping[str, Any],
    input_attrs: Mapping[Any, Any] | None = None,
    *,
    transform_prefix: str = "mt_",
) -> dict[str, Any]:
    """Provenance shared by every measure computed from one transform.

    Covers the package/version, the imported backend, the expectation type, and
    the multitaper parameters (``mt_*``) -- everything that does not depend on
    the specific measure. The per-measure attributes (``measure``,
    ``measure_kwargs_json``, and the convenient ``arg_*`` views) are added by
    the caller. Attributes on an input ``xarray.DataArray`` are carried through
    as one canonical JSON record so arbitrary keys cannot collide or produce
    invalid NetCDF attribute names.
    """
    # Namespace transform settings so they cannot collide with measure-level or
    # package-level provenance attributes.
    attrs: dict[str, Any] = {
        transform_prefix + attr: _netcdf_provenance_value(value)
        for attr, value in transform_metadata.items()
    }
    attrs["package"] = "spectral_connectivity"
    attrs["package_version"] = _package_version()
    # get_compute_backend() reports the backend actually imported (numpy vs
    # cupy), not the current env var; is_gpu_enabled() would mislabel a result if
    # SPECTRAL_CONNECTIVITY_ENABLE_GPU changed after import.
    attrs["backend"] = get_compute_backend()["backend"].upper()
    attrs["expectation_type"] = connectivity.expectation_type
    # A single fixed key is both collision-proof and a valid NetCDF attribute
    # name. Flattening arbitrary user keys would make unlike keys such as 1 and
    # "1" collide, make a structured ``x`` collide with a literal ``x_json``,
    # and let characters such as "/" create an invalid NetCDF attribute name.
    if input_attrs:
        attrs["input_attrs_json"] = _canonical_json(
            {key: _summarized_if_large(value) for key, value in input_attrs.items()}
        )
    return attrs


# Input attrs are copied onto every result variable, so an array attribute
# larger than this is recorded by shape and dtype instead of by value.
_MAX_INPUT_ATTR_ARRAY_SIZE = 100


def _summarized_if_large(value: Any) -> Any:
    """Replace an array with more than ``_MAX_INPUT_ATTR_ARRAY_SIZE`` elements."""
    if isinstance(value, (np.ndarray, list, tuple)):
        array = np.asarray(value)
        if array.size > _MAX_INPUT_ATTR_ARRAY_SIZE:
            return {
                "summarized_array": {"shape": list(array.shape), "dtype": str(array.dtype)}
            }
    return value
