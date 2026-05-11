import json
import os
import sys

import anyio
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pivotal import mcp_server


def test_get_pivotal_syntax_returns_matching_section():
    result = mcp_server.get_pivotal_syntax(topic="filter", max_chars=2000)

    assert result["ok"] is True
    assert result["topic"] == "filter"
    assert "filter" in result["content"].lower()
    json.dumps(result)


def test_get_pivotal_syntax_reports_missing_topic():
    result = mcp_server.get_pivotal_syntax(topic="definitely-not-a-topic")

    assert result["ok"] is False
    assert "No Pivotal syntax section matched" in result["message"]


def test_mcp_load_input_files_reads_csv(tmp_path):
    csv_path = tmp_path / "sales.csv"
    pd.DataFrame({"amount": [1, 2]}).to_csv(csv_path, index=False)

    inputs = mcp_server._load_input_files({"sales": str(csv_path)})

    assert list(inputs) == ["sales"]
    assert inputs["sales"].to_dict("records") == [{"amount": 1}, {"amount": 2}]


def test_compile_pivotal_source_does_not_execute():
    result = mcp_server.compile_pivotal_source(
        "with sales\n    filter amount > 0\n",
        backend="pandas",
    )

    assert result["ok"] is True
    assert result["stage"] == "codegen"
    assert "sales" in result["generated_code"]


def test_compile_pivotal_source_accepts_recent_syntax():
    result = mcp_server.compile_pivotal_source(
        (
            "with sales\n"
            "    round revenue 2 as revenue_rounded\n"
            "    rolling mean revenue 3 as rolling_revenue\n"
            "        order date\n"
            "        min_periods 1\n"
        ),
        backend="pandas",
    )

    assert result["ok"] is True
    assert "rolling(3, min_periods=1).mean()" in result["generated_code"]


def test_highlight_pivotal_source_returns_html_and_tokens():
    result = mcp_server.highlight_pivotal_source(
        'with sales\n    round revenue 2 as rounded\n',
    )

    assert result["ok"] is True
    assert '<span class="pvt-keyword">with</span>' in result["html"]
    assert '<span class="pvt-keyword">round</span>' in result["html"]
    assert '<span class="pvt-number">2</span>' in result["html"]
    assert result["css"]
    token_types = {token["text"]: token["type"] for token in result["tokens"] if token["text"].strip()}
    assert token_types["with"] == "keyword"
    assert token_types["round"] == "keyword"
    assert token_types["2"] == "number"


def test_get_pivotal_examples_documents_input_files_shape():
    result = mcp_server.get_pivotal_examples("run")

    assert result["ok"] is True
    example = result["examples"]["run"]
    assert example["arguments"]["input_files"] == {"sales": "C:\\path\\to\\sales.csv"}
    assert any("table name" in note for note in example["notes"])
    assert any("not inline CSV text" in note for note in example["notes"])


def test_create_mcp_server_requires_optional_dependency_when_missing():
    try:
        import mcp  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="pivotal-lang\\[mcp\\]"):
            mcp_server.create_mcp_server()
    else:
        server = mcp_server.create_mcp_server()
        assert hasattr(server, "run")


def test_parse_args_defaults_preserve_local_stdio():
    args = mcp_server._parse_args([])

    assert args.read_only is False
    assert args.transport == "stdio"


def test_parse_args_supports_readonly_streamable_http():
    args = mcp_server._parse_args(["--read-only", "--transport", "streamable-http"])

    assert args.read_only is True
    assert args.transport == "streamable-http"


def test_mcp_stdio_pivotal_run_does_not_deadlock(tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    csv_path = tmp_path / "sales.csv"
    pd.DataFrame({"amount": [10, -5, 20]}).to_csv(csv_path, index=False)

    async def _run():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "pivotal.mcp_server"],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert "pivotal_examples" in [tool.name for tool in tools.tools]
                result = await session.call_tool(
                    "pivotal_run",
                    {
                        "source": "with sales\n    filter amount > 0\n",
                        "input_files": {"sales": str(csv_path)},
                        "return_tables": ["sales"],
                        "timeout_seconds": 10,
                    },
                )
                return json.loads(result.content[0].text)

    result = anyio.run(_run)

    assert result["ok"] is True
    assert result["tables"]["sales"]["preview"] == [
        {"amount": 10},
        {"amount": 20},
    ]


def test_mcp_readonly_stdio_exposes_only_compile_safe_tools():
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def _run():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "pivotal.mcp_server", "--read-only"],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool_names = sorted(tool.name for tool in tools.tools)
                result = await session.call_tool(
                    "pivotal_compile",
                    {
                        "source": "with sales\n    filter amount > 0\n",
                        "backend": "pandas",
                    },
                )
                return tool_names, json.loads(result.content[0].text)

    tool_names, result = anyio.run(_run)

    assert tool_names == [
        "pivotal_compile",
        "pivotal_examples",
        "pivotal_highlight",
        "pivotal_syntax",
    ]
    assert result["ok"] is True
