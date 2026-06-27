"""Tests for intra-workflow memory isolation."""

from openfugu.conductor.memory import WorkflowMemory
from openfugu.conductor.workflow import WorkflowStep


def test_access_list_only_shares_step_outputs():
    memory = WorkflowMemory(session_id="test")
    memory.record_step_output(0, "plan", "use binary search")
    memory.record_tool_call(0, 1, {"name": "run_code"}, {"content": "secret tool output"})

    step = WorkflowStep(
        step_index=1,
        subtask="implement",
        worker_id=0,
        access_list=[0],
    )
    messages = memory.build_context_messages("original question", step)

    contents = " ".join(str(m.get("content", "")) for m in messages)
    assert "binary search" in contents
    assert "secret tool output" not in contents


def test_all_access_includes_prior_steps():
    memory = WorkflowMemory(session_id="test")
    memory.record_step_output(0, "step0", "answer0")
    memory.record_step_output(1, "step1", "answer1")

    step = WorkflowStep(
        step_index=2,
        subtask="synthesize",
        worker_id=0,
        access_list=["all"],
    )
    messages = memory.build_context_messages("question", step)
    contents = " ".join(str(m.get("content", "")) for m in messages)
    assert "answer0" in contents
    assert "answer1" in contents


def test_inter_workflow_shared_artifacts():
    memory = WorkflowMemory(session_id="test")
    memory.record_tool_call(0, 1, {"name": "read_file"}, {"content": "file contents here"})
    memory.clear_workflow_state()

    step = WorkflowStep(step_index=0, subtask="new task", worker_id=0, access_list=[])
    messages = memory.build_context_messages("question", step)
    contents = " ".join(str(m.get("content", "")) for m in messages)
    assert "file contents" in contents
