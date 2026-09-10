from tools import ToolLogger, create_calculator_tool


def test_calculator_evaluates_basic_math(tmp_path):
    logger = ToolLogger(logs_dir=str(tmp_path))
    calc = create_calculator_tool(logger)
    result = calc.invoke("22000 * 0.10")
    assert "2200" in result
    assert logger.get_logs()
    assert logger.get_logs()[0]["tool_name"] == "calculator"


def test_calculator_rejects_unsafe_expression(tmp_path):
    logger = ToolLogger(logs_dir=str(tmp_path))
    calc = create_calculator_tool(logger)
    result = calc.invoke("__import__('os').system('pwd')")
    assert "Invalid expression" in result
