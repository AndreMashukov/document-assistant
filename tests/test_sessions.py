import json
from pathlib import Path

from assistant import DocumentAssistant


def _assistant(tmp_path: Path) -> DocumentAssistant:
    assistant = DocumentAssistant(
        openai_api_key="test-key",
        session_storage_path=str(tmp_path / "sessions"),
    )
    assistant.logs_dir = str(tmp_path / "logs")
    return assistant


def test_start_session_writes_json_and_binds_logger(tmp_path):
    assistant = _assistant(tmp_path)
    session_id = assistant.start_session("demo_user", session_id="demo-session")

    session_file = tmp_path / "sessions" / "demo-session.json"
    assert session_file.exists()
    payload = json.loads(session_file.read_text())
    assert payload["session_id"] == "demo-session"
    assert payload["user_id"] == "demo_user"
    assert payload["conversation_history"] == []
    assert "created_at" in payload
    assert "last_updated" in payload

    assert session_id == "demo-session"
    assert assistant.tool_logger.session_id == "demo-session"
    assert assistant.tool_logger.log_file.endswith("session_demo-session.json")


def test_session_file_reloads_and_tools_write_session_log(tmp_path):
    assistant = _assistant(tmp_path)
    assistant.start_session("demo_user", session_id="demo-session")

    calculator = next(tool for tool in assistant.tools if tool.name == "calculator")
    reader = next(tool for tool in assistant.tools if tool.name == "document_reader")
    calculator.invoke("5000 + 12500")
    reader.invoke("INV-001")

    log_file = Path(assistant.tool_logger.log_file)
    assert log_file.exists()
    logs = json.loads(log_file.read_text())
    assert logs[0]["tool_name"] == "calculator"
    assert {"timestamp", "tool_name", "input", "output"} <= set(logs[0])
    assert logs[1]["tool_name"] == "document_reader"

    assistant.current_session.conversation_history.append({
        "user_input": "What's the total in INV-001?",
        "intent": {"intent_type": "qa", "confidence": 0.9, "reasoning": "asks for a total"},
        "tools_used": ["document_reader", "calculator"],
        "summary": "Looked up INV-001 and added line items.",
        "messages": [
            {"role": "human", "content": "What's the total in INV-001?"},
            {"role": "ai", "content": "INV-001 totals $22,000."},
        ],
    })
    assistant.current_session.document_context = ["INV-001"]
    assistant._save_session()

    reloaded = assistant._load_session("demo-session")
    assert reloaded.session_id == "demo-session"
    assert reloaded.document_context == ["INV-001"]
    assert reloaded.conversation_history[0]["messages"][0]["role"] == "human"
