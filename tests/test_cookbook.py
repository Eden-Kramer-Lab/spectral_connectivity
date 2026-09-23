"""Execute every recipe in the doctested documentation pages.

Keeps the copy-pasteable cookbook and assistant guide honest against the
current API.
"""

import doctest
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"


@pytest.mark.parametrize("page", ["cookbook.md", "llm_guide.md"])
def test_cookbook_recipes_run(page):
    """All fenced code blocks on the page execute with expected output."""
    document = DOCS / page
    assert document.exists(), f"{page} not found at {document}"
    failures, attempted = doctest.testfile(
        str(document),
        module_relative=False,
        encoding="utf-8",
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        verbose=False,
    )
    assert attempted > 0, f"no doctest examples were found in {page}"
    assert failures == 0
