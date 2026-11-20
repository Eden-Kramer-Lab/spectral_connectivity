"""Assertions shared by the tests of deprecated 2.x spellings."""

import warnings
from collections.abc import Callable
from typing import Any


def warns_once_at_the_call(call: Callable[[], Any], match: str) -> tuple[Any, str]:
    """Run ``call`` and check it raised exactly one ``DeprecationWarning``.

    ``call`` is a lambda written on the line that uses the deprecated spelling;
    the warning must contain ``match``, say "removed in 4.0", and be attributed
    to that line. Returns the call's result and the warning's message.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = call()
    deprecations = [w for w in record if issubclass(w.category, DeprecationWarning)]
    code = call.__code__
    assert [(w.filename, w.lineno) for w in deprecations] == [
        (code.co_filename, code.co_firstlineno)
    ], [str(w.message) for w in deprecations]
    message = str(deprecations[0].message)
    assert match in message
    assert "removed in 4.0" in message
    return result, message
