"""Execute every recipe in docs/cookbook.md as a doctest.

Keeps the copy-pasteable cookbook honest against the current API.
"""

import doctest
from pathlib import Path

COOKBOOK = Path(__file__).resolve().parent.parent / "docs" / "cookbook.md"


def test_cookbook_recipes_run():
    """All fenced code blocks in the cookbook execute with expected output."""
    assert COOKBOOK.exists(), f"cookbook not found at {COOKBOOK}"
    failures, _ = doctest.testfile(
        str(COOKBOOK),
        module_relative=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        verbose=False,
    )
    assert failures == 0
