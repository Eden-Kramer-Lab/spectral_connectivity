"""Write the measure table in CONNECTIVITY_METRIC_RANGES.md from list_measures().

Run ``python docs/generate_measure_table.py`` after changing a measure's
metadata; ``tests/test_docs.py`` fails while the committed table is stale.
"""

import math
from pathlib import Path

from spectral_connectivity import list_measures

DOCUMENT = Path(__file__).with_name("CONNECTIVITY_METRIC_RANGES.md")
START = "<!-- measure-table:start -->"
END = "<!-- measure-table:end -->"
FORMULA_START = "<!-- formula-table:start -->"
FORMULA_END = "<!-- formula-table:end -->"
_DIAGONALS = {
    "masked_nan": "NaN (masked)",
    "undefined_nan": "NaN (undefined)",
    "estimated": "estimated",
    "not_applicable": "none",
}


def _bound(value: float) -> str:
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    if math.isclose(abs(value), math.pi):
        return "π" if value > 0 else "-π"
    return f"{value:g}"


def _range(lower: float, upper: float, is_complex: bool) -> str:
    opening = "(" if math.isinf(lower) else "["
    closing = ")" if math.isinf(upper) else "]"
    interval = f"{opening}{_bound(lower)}, {_bound(upper)}{closing}"
    return f"\\|z\\| in {interval}" if is_complex else interval


def _diagonal(policy: str, value: float | None, nan_for_zero_power: bool) -> str:
    if value is None:
        return _DIAGONALS[policy]
    text = f"{value:g}, plus rounding noise" if policy == "rounding_noise" else f"{value:g}"
    return f"{text} (NaN for a zero-power signal)" if nan_for_zero_power else text


def _meaning(interpretation: str, sign_convention: str | None) -> str:
    return interpretation if sign_convention is None else f"{interpretation} {sign_convention}"


def measure_table() -> str:
    """The Markdown table of every measure's range, units, diagonal, and meaning."""
    rows = [
        "| Measure | Range | Units | Diagonal | Interpretation |",
        "|---|---|---|---|---|",
    ]
    rows.extend(
        f"| `{measure.name}` "
        f"| {_range(*measure.value_range, measure.is_complex)} "
        f"| {measure.units} "
        f"| {_diagonal(measure.diagonal_policy, measure.diagonal_value, measure.diagonal_nan_for_zero_power)} "
        f"| {_meaning(measure.interpretation, measure.sign_convention)} |"
        for measure in list_measures()
    )
    return "\n".join(rows)


def _table_cell(text: str) -> str:
    """``text`` with the pipes a Markdown table would split on escaped."""
    return text.replace("|", "\\|")


def formula_table() -> str:
    """The Markdown table of the coherency family's quantities and formulas."""
    rows = ["| Measure | Quantity | Formula |", "|---|---|---|"]
    rows.extend(
        f"| `{measure.name}` | {measure.quantity.replace('_', ' ')} "
        f"| {_table_cell(measure.formula)} |"
        for measure in list_measures()
        if measure.formula is not None and measure.quantity is not None
    )
    return "\n".join(rows)


def _between(document: str, start: str, end: str, table: str) -> str:
    head, rest = document.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{table}\n{end}{tail}"


def render(document: str) -> str:
    """``document`` with the text between each pair of table markers regenerated."""
    document = _between(document, START, END, measure_table())
    return _between(document, FORMULA_START, FORMULA_END, formula_table())


if __name__ == "__main__":
    DOCUMENT.write_text(render(DOCUMENT.read_text(encoding="utf-8")), encoding="utf-8")
