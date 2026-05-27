"""Fuzz the detection-record JSON parser on the Python and C++ sides.

The contract is that malformed input never crashes the parser: the Python side
raises a normal exception, and the C++ aggregator exits cleanly with a non-zero
code rather than aborting on a signal.
"""

from __future__ import annotations

import contextlib
import json
import subprocess

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fleetwatch.aggregator import aggregator_path
from fleetwatch.ingest import parse_line


@settings(max_examples=300, deadline=None)
@given(blob=st.text(max_size=200))
def test_python_parser_never_crashes(blob: str) -> None:
    with contextlib.suppress(ValueError, json.JSONDecodeError):
        parse_line(blob)  # malformed input raises a normal exception


@settings(max_examples=200, deadline=None)
@given(
    blob=st.one_of(
        st.text(max_size=200),
        st.binary(max_size=200).map(lambda b: b.decode("latin-1")),
    )
)
@pytest.mark.skipif(aggregator_path() is None, reason="aggregator not built")
def test_cpp_aggregator_never_crashes(blob: str) -> None:
    path = aggregator_path()
    assert path is not None
    proc = subprocess.run(
        [path], input=blob, capture_output=True, text=True, check=False
    )
    # Clean exit codes only: 0 (parsed), 1 (error), 2 (parse error).
    # A negative return code means a signal (crash), which must never happen.
    assert proc.returncode in (0, 1, 2), f"unexpected exit {proc.returncode}"


@pytest.mark.skipif(aggregator_path() is None, reason="aggregator not built")
def test_cpp_aggregator_rejects_structured_garbage() -> None:
    path = aggregator_path()
    assert path is not None
    bad_inputs = [
        "",
        "{",
        '{"frames":',
        '{"frames":[{"detections":[{"bbox":[1,1,0,0]}]}]}',
        '{"frames":[]}trailing',
        '{"iou_threshold":5,"frames":[]}',
    ]
    for text in bad_inputs:
        proc = subprocess.run([path], input=text, capture_output=True, text=True, check=False)
        assert proc.returncode == 2, f"expected parse-error exit for {text!r}"
