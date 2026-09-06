from verl.utils.reward_score.bc_p_llm_judge import _extract_final_response


def test_extract_final_response_accepts_tool_call_list():
    solution = """<tool_call>
    [
      {"name": "search", "arguments": {"query": "example"}},
      {"name": "finish", "arguments": {"answer": "Paris", "confidence": "90%"}}
    ]
    </tool_call>"""

    assert _extract_final_response(solution) == "Exact Answer: Paris\nConfidence: 90%"


def test_extract_final_response_ignores_non_dict_list_items():
    solution = '<tool_call>[null, "bad payload", 1]</tool_call>'

    assert _extract_final_response(solution) == ""
