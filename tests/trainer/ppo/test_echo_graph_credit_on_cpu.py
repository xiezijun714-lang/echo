import asyncio
from types import SimpleNamespace

import numpy as np
import torch

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.trainer.ppo.core_algos import compute_supo_advantage
from verl.workers.rollout.replica import TokenOutput


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _run_graph_credit(
    graph,
    *,
    gamma=0.5,
    aggregation="max",
    clip_max=1.0,
    selected_turns=None,
    negative_scale=0.0,
    return_metrics=False,
    response_turn_ids=None,
    finding_turn_ids=None,
    selection_masks=None,
):
    # One positive and one negative rollout share a uid. The positive rollout
    # has three trajectory segments; the final segment carries reward and DAG.
    response_mask = torch.ones((4, 4), dtype=torch.long)
    rewards = torch.zeros((4, 4), dtype=torch.float32)
    rewards[2, -1] = 1.0
    config = SimpleNamespace(
        echo_credit_method="graph",
        echo_neg_penalty_ratio=negative_scale,
        echo_graph_gamma=gamma,
        echo_graph_aggregation=aggregation,
        echo_graph_clip_max=clip_max,
    )
    graph_metrics = {}
    advantages, _ = compute_supo_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["uid", "uid", "uid", "uid"], dtype=object),
        rollout_id=np.array(["positive", "positive", "positive", "negative"], dtype=object),
        is_final=np.array([False, False, True, True]),
        overlong=np.array([False, False, False, False]),
        traj_idx=np.array([0, 1, 2, 0]),
        echo_selected_turn_ids=_object_array([None, None, selected_turns or [], []]),
        echo_memory_graph=_object_array([
            None,
            None,
            graph,
            {"version": 1, "nodes": [], "outcome_parent_turn_ids": []},
        ]),
        echo_response_turn_ids=_object_array(
            response_turn_ids or [
                [0, 1, 3, 3],
                [2, 2, 4, 4],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ]
        ),
        echo_response_finding_turn_ids=_object_array(
            finding_turn_ids or [[-1, -1, -1, -1]] * 4
        ),
        echo_response_selection_mask=_object_array(selection_masks or [[0, 0, 0, 0]] * 4),
        norm_adv_by_std_in_grpo=False,
        config=config,
        echo_graph_metrics=graph_metrics,
    )
    return (advantages, graph_metrics) if return_metrics else advantages


def test_echo_graph_credit_chain_and_recent_only_turn():
    graph = {
        "version": 1,
        "nodes": [
            {"turn_id": 0, "parent_turn_ids": []},
            {"turn_id": 1, "parent_turn_ids": [0]},
            {"turn_id": 2, "parent_turn_ids": [1]},
            # Turn 3 was retained automatically but never selected by the model.
            {"turn_id": 3, "parent_turn_ids": []},
            # Turn 4 is a later, unrelated child of turn 0 and must not leak credit.
            {"turn_id": 4, "parent_turn_ids": [0]},
        ],
        "outcome_parent_turn_ids": [2],
    }

    advantages = _run_graph_credit(graph, gamma=0.5)

    assert torch.allclose(advantages[0], torch.tensor([0.0625, 0.125, 0.0, 0.0]))
    assert torch.allclose(advantages[1], torch.tensor([0.25, 0.25, 0.0, 0.0]))
    assert torch.allclose(advantages[2], torch.full((4,), 0.5))
    assert torch.count_nonzero(advantages[3]) == 0


def test_echo_negative_advantage_is_dense_and_scaled():
    graph = {
        "version": 1,
        "nodes": [
            {"turn_id": 0, "parent_turn_ids": []},
        ],
        "outcome_parent_turn_ids": [0],
    }

    advantages = _run_graph_credit(graph, negative_scale=0.5)

    # The negative rollout is not assigned a causal turn path, so its signed
    # group advantage is applied densely to every trainable response token.
    assert torch.allclose(advantages[3], torch.full((4,), -0.25))
    # Positive graph credit remains unchanged by the negative-update option.
    assert torch.allclose(advantages[0], torch.tensor([0.25, 0.0, 0.0, 0.0]))


def test_echo_graph_credit_accumulates_multiple_paths_and_clips():
    graph = {
        "version": 1,
        "nodes": [
            {"turn_id": 0, "parent_turn_ids": []},
            {"turn_id": 1, "parent_turn_ids": [0]},
            {"turn_id": 2, "parent_turn_ids": [0]},
            {
                "turn_id": 4,
                "parent_turn_ids": [1, 2],
            },
        ],
        "outcome_parent_turn_ids": [4],
    }

    advantages, metrics = _run_graph_credit(
        graph,
        gamma=1.0,
        aggregation="sum",
        clip_max=1.0,
        return_metrics=True,
    )

    # Turn 0 receives two path contributions, but the coefficient is clipped to 1.
    assert torch.allclose(advantages[0], torch.tensor([0.5, 0.5, 0.0, 0.0]))
    assert torch.allclose(advantages[1], torch.tensor([0.5, 0.5, 0.5, 0.5]))
    assert metrics["clip_ratio"] == 0.25
    assert metrics["branch_parent_weight_mean"] == 1.0

    no_clip_advantages, no_clip_metrics = _run_graph_credit(
        graph,
        gamma=1.0,
        aggregation="sum",
        clip_max=None,
        return_metrics=True,
    )
    assert no_clip_advantages[0, 0] == 1.0
    assert no_clip_metrics["clip_ratio"] == 0.0
    assert no_clip_metrics["branch_parent_weight_mean"] == 2.0


def test_echo_graph_uses_one_selection_level_discount():
    graph = {
        "version": 1,
        "nodes": [
            {"turn_id": 0, "parent_turn_ids": []},
            {"turn_id": 1, "parent_turn_ids": [0]},
            {"turn_id": 2, "parent_turn_ids": [1]},
        ],
        "outcome_parent_turn_ids": [2],
    }

    advantages = _run_graph_credit(
        graph,
        gamma=0.5,
        clip_max=None,
    )

    assert torch.allclose(advantages[0], torch.tensor([0.0625, 0.125, 0.0, 0.0]))
    assert torch.allclose(advantages[1], torch.tensor([0.25, 0.25, 0.0, 0.0]))


def test_echo_graph_auxiliary_actions_inherit_credit_outside_the_dag():
    graph = {
        "version": 1,
        "nodes": [
            {"turn_id": 0, "parent_turn_ids": []},
            {
                "turn_id": 1,
                "parent_turn_ids": [0, 3],
            },
            {"turn_id": 2, "parent_turn_ids": [1]},
            {"turn_id": 3, "parent_turn_ids": []},
        ],
        "outcome_parent_turn_ids": [2],
    }

    advantages = _run_graph_credit(
        graph,
        gamma=0.5,
        clip_max=None,
        finding_turn_ids=[[-1, -1, -1, 0], [-1, -1, -1, -1], [-1] * 4, [-1] * 4],
        response_turn_ids=[[0, 1, 3, -1], [2, 2, 4, 4], [-1] * 4, [-1] * 4],
        selection_masks=[[0, 0, 0, 0], [0, 0, 1, 1], [0] * 4, [0] * 4],
    )

    # Selection tokens are trained independently with unit weight; they do
    # not consume or inherit graph credit from the selected parent turns.
    assert torch.allclose(advantages[1], torch.tensor([0.25, 0.25, 0.5, 0.5]))
    # Finding tokens inherit the graph credit of their source turn.
    assert advantages[0, 3] == 0.0625


def test_echo_graph_zero_gamma_blocks_historical_credit():
    graph = {
        "version": 1,
        "nodes": [
            {"turn_id": 0, "parent_turn_ids": []},
            {"turn_id": 1, "parent_turn_ids": [0]},
            {"turn_id": 2, "parent_turn_ids": [1]},
        ],
        "outcome_parent_turn_ids": [2],
    }

    advantages = _run_graph_credit(graph, gamma=0.0)

    assert torch.count_nonzero(advantages[0]) == 0
    assert torch.count_nonzero(advantages[1]) == 0


def test_echo_graph_cycle_falls_back_to_token_credit():
    cyclic_graph = {
        "version": 1,
        "nodes": [
            {"turn_id": 0, "parent_turn_ids": [1]},
            {"turn_id": 1, "parent_turn_ids": [0]},
        ],
        "outcome_parent_turn_ids": [1],
    }

    advantages = _run_graph_credit(cyclic_graph, selected_turns=[1])

    assert torch.allclose(advantages[0], torch.tensor([0.0, 0.5, 0.0, 0.0]))
    assert torch.count_nonzero(advantages[1]) == 0
    assert torch.allclose(advantages[2], torch.full((4,), 0.5))


def test_echo_selection_keeps_model_choices_separate_from_recent_context():
    loop = object.__new__(ToolAgentLoop)
    loop.selection_max_turns = 8
    loop.echo_recent_turns = 2

    model_selected = loop._parse_selection_indices("<selection>\nturn_0\n</selection>", 4)
    active_context = loop._merge_selected_with_recent(model_selected, 4)

    assert [item["index"] for item in model_selected] == [0]
    assert [item["index"] for item in active_context] == [0, 2, 3]
    assert loop._parse_selection_indices("<selection></selection>", 4) == []
    assert [item["index"] for item in loop._parse_selection_indices(
        "<selection>turn_0, turn_2</selection>", 4
    )] == [0, 2]
    assert loop._parse_selection_indices("selection unavailable", 4) is None


def test_echo_graph_parent_set_includes_automatic_recent_context():
    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.current_selected_turn_ids = [2, 5]
    agent_data.current_model_selected_turn_ids = [2]

    assert ToolAgentLoop._current_echo_parent_turn_ids(agent_data) == [2, 5]


def test_echo_memory_graph_serializes_completed_turns_and_explicit_parents():
    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.turn_history = [
        {"turn_id": 0, "source_traj_idx": 0, "parent_turn_ids": []},
        {
            "turn_id": 1,
            "source_traj_idx": 1,
            "parent_turn_ids": [0],
        },
    ]
    # A pending Turn is intentionally not part of turn_history and must not be serialized.
    agent_data.pending_turn = {"turn_id": 2, "parent_turn_ids": [1]}
    agent_data.current_segment_turn_ids = [1]
    graph = ToolAgentLoop._build_echo_memory_graph(agent_data)

    assert graph == {
        "version": 1,
        "nodes": [
            {"turn_id": 0, "source_traj_idx": 0, "parent_turn_ids": []},
            {
                "turn_id": 1,
                "source_traj_idx": 1,
                "parent_turn_ids": [0],
            },
        ],
        "outcome_parent_turn_ids": [1],
        "final_model_selected_turn_ids": [],
    }


def test_echo_parent_ids_include_active_selection_and_current_segment():
    loop = object.__new__(ToolAgentLoop)
    loop.sum_last_turn_max_chars = 100
    loop._build_turn_text = lambda action, finding: f"{action}: {finding}"
    loop._mark_finding_turn_tokens = lambda *_args: None

    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.turn_history = [
        {"turn_id": 2, "source_traj_idx": 0, "parent_turn_ids": []},
        {"turn_id": 5, "source_traj_idx": 0, "parent_turn_ids": []},
    ]
    agent_data.current_selected_turn_ids = [2, 5]
    agent_data.current_model_selected_turn_ids = [2, 5]
    expected_parent_ids_by_turn = {
        6: [2, 5],
        7: [2, 5, 6],
        8: [2, 5, 6, 7],
    }
    for turn_id, expected_parent_ids in expected_parent_ids_by_turn.items():
        assert ToolAgentLoop._current_echo_parent_turn_ids(agent_data) == expected_parent_ids
        agent_data.pending_turn = {
            "turn_id": turn_id,
            "action": f"search({turn_id})",
            "finding": f"finding-{turn_id}",
            "parent_turn_ids": expected_parent_ids,
        }
        assert loop._finalize_pending_echo_turn(agent_data, "")

    graph = ToolAgentLoop._build_echo_memory_graph(agent_data)
    assert graph["outcome_parent_turn_ids"] == [2, 5, 6, 7, 8]
    assert [node["parent_turn_ids"] for node in graph["nodes"][-3:]] == [
        [2, 5],
        [2, 5, 6],
        [2, 5, 6, 7],
    ]


def test_echo_prompt_keeps_overflow_turn_as_local_context():
    loop = object.__new__(ToolAgentLoop)
    loop.sum_last_turn_hint = "record the latest result"

    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.original_messages = [
        {"role": "system", "content": "Solve the task."},
        {"role": "user", "content": "What is the answer?"},
    ]
    agent_data.turn_history = [
        {"turn_id": 0, "action": "search(query=old)", "finding": "old fact"},
    ]
    # This is the action/observation pair that caused the boundary. It is not
    # a completed source-indexed turn until the next assistant finding exists.
    agent_data.echo_pending_local_turn = {
        "turn_id": 1,
        "action": "search(query=current)",
        "finding": "",
    }
    agent_data.echo_pending_local_messages = [{"role": "tool", "content": "current fact"}]

    prompt = loop._build_echo_prompt(agent_data, [{"index": 0}])
    contents = [message["content"] for message in prompt]

    assert "[turn_0 Key Finding] old fact" in contents
    assert "[current turn Action] search(query=current)" in contents
    assert "current fact" in contents
    assert contents[-2] == "record the latest result"


def test_echo_overflow_rollback_preserves_previous_turn_finding_tokens():
    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.response_ids = [3, 4, 5]
    agent_data.accumulated_response_ids = [10, 11, 12, 13, 14]
    agent_data.accumulated_response_mask = [1, 1, 1, 1, 1]
    agent_data.accumulated_logprobs = [0.1, 0.2, 0.3, 0.4, 0.5]
    agent_data.accumulated_response_turn_ids = [0, 0, 1, 1, 1]
    agent_data.accumulated_response_finding_turn_ids = [-1, -1, -1, 7, -1]
    agent_data.accumulated_response_selection_mask = [0, 0, 0, 0, 0]

    ToolAgentLoop._rollback_current_echo_response(agent_data)

    assert agent_data.accumulated_response_ids == [10, 11, 13]
    assert agent_data.accumulated_response_turn_ids == [0, 0, 1]
    assert agent_data.accumulated_response_finding_turn_ids == [-1, -1, 7]
    assert agent_data.accumulated_response_mask == [1, 1, 1]


def test_echo_overflow_segment_keeps_the_action_that_caused_boundary():
    loop = object.__new__(ToolAgentLoop)
    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.current_traj_prompt_ids = [1, 2]
    agent_data.accumulated_response_ids = [10, 11, 12]
    agent_data.accumulated_response_mask = [1, 1, 0]
    agent_data.accumulated_logprobs = [0.1, 0.2, 0.0]
    agent_data.accumulated_response_turn_ids = [0, 0, 0]
    agent_data.accumulated_response_finding_turn_ids = [-1, -1, -1]
    agent_data.accumulated_response_selection_mask = [0, 0, 0]

    loop._save_current_trajectory_without_current_turn(agent_data)

    assert len(agent_data.trajectory_outputs) == 1
    assert agent_data.trajectory_outputs[0].response_ids == [10, 11, 12]
    assert agent_data.trajectory_outputs[0].response_mask == [1, 1, 0]


def test_tool_response_budget_keeps_observation_before_terminating():
    ids, exhausted = ToolAgentLoop._fit_tool_response_to_budget(
        [1, 2, 3, 4], response_budget_length=6, response_length=8, is_finish_tool=False
    )
    assert ids == [1, 2]
    assert exhausted is True

    ids, exhausted = ToolAgentLoop._fit_tool_response_to_budget(
        [1, 2], response_budget_length=8, response_length=8, is_finish_tool=False
    )
    assert ids == []
    assert exhausted is True

    ids, exhausted = ToolAgentLoop._fit_tool_response_to_budget(
        [1, 2], response_budget_length=8, response_length=8, is_finish_tool=True
    )
    assert ids == []
    assert exhausted is True


def test_full_response_budget_still_dispatches_a_valid_tool_call():
    class FakeServer:
        async def generate(self, **_kwargs):
            return TokenOutput(token_ids=[1, 2], log_probs=[-0.1, -0.2], stop_reason="completed")

    class FakeParser:
        async def extract_tool_calls(self, _response_ids):
            return "", [FunctionCall(name="search", arguments='{"query":"echo"}')]

    loop = object.__new__(ToolAgentLoop)
    loop.server_manager = FakeServer()
    loop.tool_parser = FakeParser()
    loop.enable_summarization = False
    loop.max_assistant_turns = 10
    loop.max_user_turns = None
    loop.response_length = 2
    loop.interaction_config_file = None

    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.prompt_ids = [99]

    state = asyncio.run(loop._handle_generating_state(agent_data, {}))

    assert state is AgentState.PROCESSING_TOOLS
    assert len(agent_data.tool_calls) == 1
    assert agent_data.tool_calls[0].name == "search"


def test_last_assistant_turn_dispatches_tool_call_before_terminating():
    class FakeServer:
        async def generate(self, **_kwargs):
            return TokenOutput(token_ids=[1, 2], log_probs=[-0.1, -0.2], stop_reason="completed")

    class FakeParser:
        async def extract_tool_calls(self, _response_ids):
            return "", [FunctionCall(name="finish", arguments='{"answer":"done"}')]

    loop = object.__new__(ToolAgentLoop)
    loop.server_manager = FakeServer()
    loop.tool_parser = FakeParser()
    loop.enable_summarization = False
    loop.max_assistant_turns = 1
    loop.max_user_turns = None
    loop.response_length = 8
    loop.interaction_config_file = None

    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.prompt_ids = [99]

    state = asyncio.run(loop._handle_generating_state(agent_data, {}))

    assert state is AgentState.PROCESSING_TOOLS
    assert agent_data.assistant_turns == 1
    assert agent_data.tool_calls[0].name == "finish"


def test_tool_response_terminates_after_last_assistant_turn():
    class FakeParser:
        async def extract_tool_calls(self, _response_ids):
            return "", []

    loop = object.__new__(ToolAgentLoop)
    loop.tool_parser_name = "hermes"
    loop.max_parallel_calls = 1
    loop.max_assistant_turns = 1
    loop.max_user_turns = None
    loop.enable_summarization = False
    loop.response_length = 8
    loop.interaction_config_file = None

    async def fake_apply_chat_template(_messages, **_kwargs):
        return [7, 8]

    async def fake_call_tool(_tool_call, _tools_kwargs, _agent_data):
        return type("ToolResponseStub", (), {"text": "observation", "image": None, "video": None})(), 0.0, {}

    loop.apply_chat_template = fake_apply_chat_template
    loop._call_tool = fake_call_tool

    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.assistant_turns = 1
    agent_data.tool_calls = [FunctionCall(name="finish", arguments='{"answer":"done"}')]
    agent_data.prompt_ids = [1, 2]
    agent_data.response_ids = [3, 4]
    agent_data.response_mask = [1, 1]
    agent_data.response_logprobs = [-0.1, -0.2]
    loop.tool_parser = FakeParser()

    state = asyncio.run(loop._handle_processing_tools_state(agent_data))

    assert state is AgentState.TERMINATED
    assert agent_data.user_turns == 1
    assert agent_data.prompt_ids[-2:] == [7, 8]


def test_echo_overflow_on_last_user_turn_does_not_enter_selector():
    loop = object.__new__(ToolAgentLoop)
    loop.tool_parser_name = "hermes"
    loop.max_parallel_calls = 1
    loop.max_assistant_turns = 100
    loop.max_user_turns = 1
    loop.enable_summarization = True
    loop.context_compression_method = "echo_e2e"
    loop.response_length = 100
    loop.working_context_length = 1
    loop.max_summary_rounds = 5
    loop.sum_last_turn_hint = "hint"
    loop.sum_last_turn_max_chars = 300
    loop.interaction_config_file = None
    loop.tokenizer = type("TokenizerStub", (), {"decode": lambda self, _ids, **_kwargs: ""})()

    async def fake_apply_chat_template(_messages, **_kwargs):
        return [7, 8]

    async def fake_call_tool(_tool_call, _tools_kwargs, _agent_data):
        return type("ToolResponseStub", (), {"text": "observation", "image": None, "video": None})(), 0.0, {}

    async def fail_selector(_agent_data):
        raise AssertionError("last user turn must terminate after its tool response")

    loop.apply_chat_template = fake_apply_chat_template
    loop._call_tool = fake_call_tool
    loop._trigger_echo_selection = fail_selector

    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.current_traj_prompt_ids = [1]
    agent_data.prompt_ids = [1, 2]
    agent_data.response_ids = [3, 4]
    agent_data.response_mask = [1, 1]
    agent_data.response_logprobs = [-0.1, -0.2]
    agent_data.tool_calls = [FunctionCall(name="search", arguments='{"query":"echo"}')]

    state = asyncio.run(loop._handle_processing_tools_state(agent_data))

    assert state is AgentState.TERMINATED
    assert agent_data.user_turns == 1
    assert agent_data.prompt_ids[-2:] == [7, 8]


def test_echo_selector_is_saved_when_first_overflow_has_no_action_segment():
    loop = object.__new__(ToolAgentLoop)
    loop.context_compression_method = "echo_e2e"
    loop.prompt_length = 4
    loop.response_length = 8

    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.current_traj_prompt_ids = [1, 2, 3, 4]
    agent_data.accumulated_response_ids = []
    agent_data.sum_instruction_ids = [5, 6]
    agent_data.response_ids = [7, 8]

    loop._append_summary_to_current_trajectory(agent_data, [0.7, 0.8])

    assert len(agent_data.trajectory_outputs) == 1
    trajectory = agent_data.trajectory_outputs[0]
    assert trajectory.prompt_ids == [1, 2, 3, 4]
    assert trajectory.response_ids == [5, 6, 7, 8]
    assert trajectory.response_mask == [0, 0, 1, 1]
    assert trajectory.response_logprobs == [0.0, 0.0, 0.7, 0.8]


def test_supo_summary_token_training_is_configurable():
    loop = object.__new__(ToolAgentLoop)
    loop.context_compression_method = "summary"
    loop.prompt_length = 4
    loop.response_length = 8

    for train_summary_tokens, expected_summary_mask in ((True, 1), (False, 0)):
        loop.train_summary_tokens = train_summary_tokens
        agent_data = AgentData([], [], [], {}, "request", {})
        agent_data.current_traj_prompt_ids = [1, 2, 3, 4]
        agent_data.accumulated_response_ids = []
        agent_data.sum_instruction_ids = [5, 6]
        agent_data.response_ids = [7, 8]

        loop._append_summary_to_current_trajectory(agent_data, [0.7, 0.8])

        trajectory = agent_data.trajectory_outputs[0]
        assert trajectory.response_ids == [5, 6, 7, 8]
        assert trajectory.response_mask == [0, 0, expected_summary_mask, expected_summary_mask]


def test_echo_selector_reserves_response_capacity_for_selection_tokens():
    loop = object.__new__(ToolAgentLoop)
    loop.context_compression_method = "echo_e2e"
    loop.prompt_length = 2
    loop.response_length = 4

    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.current_traj_prompt_ids = [1, 2]
    agent_data.accumulated_response_ids = list(range(10, 20))
    agent_data.sum_instruction_ids = [20, 21]
    agent_data.response_ids = [30, 31]

    loop._append_summary_to_current_trajectory(agent_data, [0.3, 0.4])

    trajectory = agent_data.trajectory_outputs[0]
    assert trajectory.response_ids[-2:] == [30, 31]
    assert trajectory.response_mask[-2:] == [1, 1]
    assert len(trajectory.response_ids) <= loop.response_length
    assert agent_data.echo_prompt_budget_overflow_count == 1


def test_echo_selector_physical_cap_preserves_current_prompt():
    loop = object.__new__(ToolAgentLoop)
    loop.selection_instruction = "{turn_list}"
    loop.selection_max_turns = 8
    loop.echo_recent_turns = 3
    loop.sum_last_turn_hint = "hint"
    loop.apply_chat_template_kwargs = {}
    loop.max_model_len = 20
    loop.selection_max_new_tokens = 4

    async def fake_apply_chat_template(messages, **_kwargs):
        assert messages[0]["role"] == "user"
        return list(range(100, 108))

    loop.apply_chat_template = fake_apply_chat_template
    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.current_traj_prompt_ids = [1, 2, 3, 4, 5]
    agent_data.accumulated_response_ids = list(range(10, 40))

    asyncio.run(loop._trigger_echo_selection(agent_data))

    assert agent_data.echo_selection_prefix_ids[:5] == [1, 2, 3, 4, 5]
    assert len(agent_data.prompt_ids) <= loop.max_model_len - loop.selection_max_new_tokens - 1


def test_echo_selector_reserves_configured_decoder_budget_near_tensor_limit():
    loop = object.__new__(ToolAgentLoop)
    loop.selection_instruction = "{turn_list}"
    loop.selection_max_turns = 8
    loop.selection_max_new_tokens = 512
    loop.echo_recent_turns = 3
    loop.apply_chat_template_kwargs = {}
    loop.prompt_length = 4096
    loop.response_length = 32768
    loop.max_model_len = 40960

    async def fake_apply_chat_template(_messages, **_kwargs):
        return list(range(1000))

    loop.apply_chat_template = fake_apply_chat_template
    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.current_traj_prompt_ids = list(range(4096))
    agent_data.accumulated_response_ids = list(range(40000))

    asyncio.run(loop._trigger_echo_selection(agent_data))

    assert len(agent_data.prompt_ids) <= 4096 + 32768 - 512
    assert len(agent_data.prompt_ids) <= 40960 - 512 - 1


def test_echo_selector_sets_an_explicit_max_new_tokens_without_mutating_rollout_params():
    loop = object.__new__(ToolAgentLoop)
    loop.enable_summarization = True
    loop.context_compression_method = "echo_e2e"
    loop.selection_max_new_tokens = 512
    loop.response_length = 32768
    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.is_summarizing = True
    sampling_params = {"temperature": 0.0, "logprobs": True}

    selector_params = loop._sampling_params_for_generation(sampling_params, agent_data)

    assert selector_params["max_new_tokens"] == 512
    assert "max_new_tokens" not in sampling_params


def test_echo_generation_prompt_limit_matches_rollout_tensor_budget():
    loop = object.__new__(ToolAgentLoop)
    loop.max_model_len = 40960
    loop.prompt_length = 4096
    loop.response_length = 32768

    assert loop._generation_prompt_limit() == 36864


def test_echo_natural_termination_finalizes_pending_turn():
    loop = object.__new__(ToolAgentLoop)
    loop.sum_last_turn_max_chars = 300
    loop._mark_finding_turn_tokens = lambda *_args: None

    agent_data = AgentData([], [], [], {}, "request", {})
    agent_data.pending_turn = {
        "turn_id": 4,
        "action": "search(query=current)",
        "finding": "",
        "observation": "fallback observation",
    }

    assert loop._finalize_pending_echo_turn(
        agent_data,
        "<sum_last_turn>verified current result</sum_last_turn> final answer",
    )
    assert agent_data.pending_turn is None
    assert agent_data.turn_history[0]["turn_id"] == 4
    assert agent_data.turn_history[0]["finding"] == "verified current result"


def test_echo_token_credit_does_not_dense_credit_empty_final_selection():
    response_mask = torch.ones((3, 4), dtype=torch.long)
    rewards = torch.zeros((3, 4), dtype=torch.float32)
    rewards[1, -1] = 1.0
    config = SimpleNamespace(
        echo_credit_method="token",
        echo_graph_gamma=0.5,
        echo_graph_aggregation="max",
        echo_graph_clip_max=1.0,
    )

    advantages, _ = compute_supo_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["uid", "uid", "uid"], dtype=object),
        rollout_id=np.array(["positive", "positive", "negative"], dtype=object),
        is_final=np.array([False, True, True]),
        overlong=np.array([False, False, False]),
        traj_idx=np.array([0, 1, 0]),
        echo_selected_turn_ids=_object_array([None, [], []]),
        echo_response_turn_ids=_object_array([[0, 0, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1]]),
        echo_response_finding_turn_ids=_object_array([[-1, -1, -1, -1]] * 3),
        echo_response_selection_mask=_object_array([[0, 0, 0, 0]] * 3),
        norm_adv_by_std_in_grpo=False,
        config=config,
    )

    assert torch.count_nonzero(advantages[0]) == 0
    assert torch.count_nonzero(advantages[1]) > 0
