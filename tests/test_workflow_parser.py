"""Tests for conductor workflow parser."""

from openfugu.conductor.parser import is_well_formed, parse_workflow

SAMPLE_OUTPUT = """
Let me plan this problem.

model_id = [2, 0]
subtasks = ["Develop an efficient algorithm to count complete subarrays", "Implement in Python"]
access_list = [[], ["all"]]
"""


def test_parse_valid_workflow():
    wf = parse_workflow(SAMPLE_OUTPUT)
    assert wf is not None
    assert wf.num_steps == 2
    assert wf.steps[0].worker_id == 2
    assert wf.steps[1].access_list == ["all"]


def test_is_well_formed():
    assert is_well_formed(SAMPLE_OUTPUT) is True
    assert is_well_formed("no lists here") is False


def test_parse_mismatched_lengths():
    bad = 'model_id = [0]\nsubtasks = ["a", "b"]\naccess_list = [[]]'
    assert parse_workflow(bad) is None


def test_max_steps_truncation():
    long = (
        "model_id = [0, 1, 2, 3, 4, 5]\n"
        'subtasks = ["a", "b", "c", "d", "e", "f"]\n'
        "access_list = [[], [], [], [], [], []]"
    )
    wf = parse_workflow(long, max_steps=3)
    assert wf is not None
    assert wf.num_steps == 3
