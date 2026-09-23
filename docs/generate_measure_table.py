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

_ORIENTATION = {
    "target_source": "`[..., i, j]` is `j -> i`",
    "source_target": "`[..., i, j]` is `i` relative to `j`",
    None: "",
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


def measure_table() -> str:
    """The Markdown table of every measure's range, units, and meaning."""
    rows = [
        "| Measure | Range | Units | Low-level orientation | Interpretation |",
        "|---|---|---|---|---|",
    ]
    rows.extend(
        f"| `{measure.name}` "
        f"| {_range(*measure.value_range, measure.is_complex)} "
        f"| {measure.units} "
        f"| {_ORIENTATION[measure.array_orientation]} "
        f"| {measure.interpretation} |"
        for measure in list_measures()
    )
    return "\n".join(rows)


def render(document: str) -> str:
    """``document`` with the text between the table markers regenerated."""
    head, rest = document.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{measure_table()}\n{END}{tail}"


if __name__ == "__main__":
    DOCUMENT.write_text(render(DOCUMENT.read_text(encoding="utf-8")), encoding="utf-8")
