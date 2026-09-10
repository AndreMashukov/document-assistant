from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from agent import INTENT_TO_NODE, classify_intent, create_workflow, should_continue
from prompts import (
    CALCULATION_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    get_chat_prompt_template,
)
from schemas import UserIntent


def test_chat_prompt_selects_system_prompt():
    qa = get_chat_prompt_template("qa")
    summary = get_chat_prompt_template("summarization")
    calc = get_chat_prompt_template("calculation")
    unknown = get_chat_prompt_template("unknown")

    assert QA_SYSTEM_PROMPT in qa.messages[0].prompt.template
    assert SUMMARIZATION_SYSTEM_PROMPT in summary.messages[0].prompt.template
    assert CALCULATION_SYSTEM_PROMPT in calc.messages[0].prompt.template
    assert QA_SYSTEM_PROMPT in unknown.messages[0].prompt.template
    assert "calculator tool" in CALCULATION_SYSTEM_PROMPT.lower()


def test_intent_routing_map():
    assert INTENT_TO_NODE["qa"] == "qa_agent"
    assert INTENT_TO_NODE["summarization"] == "summarization_agent"
    assert INTENT_TO_NODE["calculation"] == "calculation_agent"


def test_should_continue_reads_next_step():
    assert should_continue({"next_step": "qa_agent"}) == "qa_agent"
    assert should_continue({}) == "end"


def test_classify_intent_sets_next_step():
    intent = UserIntent(
        intent_type="summarization",
        confidence=0.91,
        reasoning="User asked for a summary.",
    )
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = intent
    state = classify_intent(
        {"user_input": "Summarize CON-001", "messages": [HumanMessage(content="hi")]},
        {"configurable": {"llm": llm}},
    )
    assert state["next_step"] == "summarization_agent"
    assert state["intent"].intent_type == "summarization"
    assert state["actions_taken"] == ["classify_intent"]


def test_classify_intent_defaults_unknown_to_qa_agent():
    intent = UserIntent(intent_type="unknown", confidence=0.2, reasoning="unclear")
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = intent
    state = classify_intent(
        {"user_input": "hmm", "messages": []},
        {"configurable": {"llm": llm}},
    )
    assert state["next_step"] == "qa_agent"


def test_workflow_has_expected_nodes_and_checkpointer():
    graph = create_workflow(llm=None, tools=[])
    node_names = set(graph.get_graph().nodes)
    for name in (
        "classify_intent",
        "qa_agent",
        "summarization_agent",
        "calculation_agent",
        "update_memory",
    ):
        assert name in node_names
    assert graph.checkpointer is not None
