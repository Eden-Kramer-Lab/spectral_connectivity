"""Generated documentation must match the code it is generated from."""

import importlib.util
from pathlib import Path

DOCS = Path(__file__).parents[1] / "docs"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_measure_table", DOCS / "generate_measure_table.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_measure_table_is_current():
    """Regenerate with ``python docs/generate_measure_table.py`` if this fails."""
    generator = _load_generator()
    document = generator.DOCUMENT.read_text(encoding="utf-8")
    assert document == generator.render(document)
