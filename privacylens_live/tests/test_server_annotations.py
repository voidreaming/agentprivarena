"""Smoke tests verifying MCP server tool annotations.

Each read/discovery tool must have ``readOnlyHint=True`` so the SDK can treat
the call as non-mutating. Privacy extraction is narrower and is controlled by
tool privacy semantics, so search/list/get observations can stay read-only
without becoming information-flow evidence. Write/mutate tools must NOT have
this hint.

If a new tool is added without the correct annotation, these tests
catch the regression before a full 493-task run wastes hours.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from privacylens_live.runner.tool_registry import (
    SERVICE_BY_SERVER_MODULE,
    read_tools_for_server_module,
    write_tools_for_server_module,
)


# The MCP servers live outside the normal package tree and use relative
# imports from ``base.py``. Add the directory to sys.path so Python
# can resolve ``from base import ...`` in the server modules.
MCP_DIR = Path(__file__).resolve().parent.parent / "mcp_servers"
if str(MCP_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_DIR))


SERVER_MODULES = tuple(SERVICE_BY_SERVER_MODULE)

SEARCH_PARAMETER_DESCRIPTIONS = (
    ("bookstack_server", "search_pages", "query", True),
    ("google_drive_server", "search_files", "query", False),
    ("mailpit_server", "search_emails", "query", True),
    ("mailpit_server", "list_contacts", "name", True),
    ("mattermost_server", "search_messages", "query", True),
    ("radicale_server", "search_events", "query", True),
    ("rocketchat_server", "search_messages", "query", True),
    ("rocketchat_server", "search_users", "query", True),
    ("gotosocial_server", "search_users", "query", True),
    ("gotosocial_server", "search_posts", "query", True),
)

LEGACY_CONTENT_ACCESS_TOOLS = {
    "bookstack_server": {"get_page"},
    "mattermost_server": {"list_messages"},
    "rocketchat_server": {"get_channel_history"},
    "gotosocial_server": {"get_profile", "list_user_posts"},
    "radicale_server": {"get_event"},
}


def _list_tools(module_name: str):
    """Import a server module and list its FastMCP tools."""
    import importlib

    mod = importlib.import_module(module_name)
    return asyncio.run(mod.mcp.list_tools())


def _tool_by_name(module_name: str, tool_name: str):
    tools = _list_tools(module_name)
    tool_map = {tool.name: tool for tool in tools}
    assert tool_name in tool_map, (
        f"{module_name}: expected tool '{tool_name}' not found. "
        f"Available: {sorted(tool_map.keys())}"
    )
    return tool_map[tool_name]


@pytest.mark.parametrize(
    "server_module",
    SERVER_MODULES,
    ids=SERVER_MODULES,
)
def test_readonly_tools_have_annotation(server_module: str):
    """Read/discovery tools must declare readOnlyHint=True."""
    tools = _list_tools(server_module)
    tool_map = {t.name: t for t in tools}
    expected = read_tools_for_server_module(server_module)

    for tool_name in expected:
        assert tool_name in tool_map, (
            f"{server_module}: expected tool '{tool_name}' not found. "
            f"Available: {sorted(tool_map.keys())}"
        )
        tool = tool_map[tool_name]
        assert tool.annotations is not None, (
            f"{server_module}.{tool_name}: missing annotations "
            f"(need readOnlyHint=True for privacy extraction)"
        )
        assert tool.annotations.readOnlyHint is True, (
            f"{server_module}.{tool_name}: readOnlyHint is "
            f"{tool.annotations.readOnlyHint!r}, expected True"
        )


@pytest.mark.parametrize(
    "server_module",
    [module for module in SERVER_MODULES if write_tools_for_server_module(module)],
    ids=[module for module in SERVER_MODULES if write_tools_for_server_module(module)],
)
def test_write_tools_do_not_have_readonly_annotation(server_module: str):
    """Write/mutate tools must NOT have readOnlyHint=True."""
    tools = _list_tools(server_module)
    tool_map = {t.name: t for t in tools}
    expected = write_tools_for_server_module(server_module)

    for tool_name in expected:
        assert tool_name in tool_map, (
            f"{server_module}: expected tool '{tool_name}' not found. "
            f"Available: {sorted(tool_map.keys())}"
        )
        tool = tool_map[tool_name]
        has_readonly = (
            tool.annotations is not None and tool.annotations.readOnlyHint is True
        )
        assert not has_readonly, (
            f"{server_module}.{tool_name}: write tool should NOT have readOnlyHint=True"
        )


@pytest.mark.parametrize(
    "server_module",
    SERVER_MODULES,
    ids=SERVER_MODULES,
)
def test_no_unexpected_tools(server_module: str):
    """Every tool must be in EXPECTED_READONLY or EXPECTED_WRITE."""
    tools = _list_tools(server_module)
    known = read_tools_for_server_module(server_module) | write_tools_for_server_module(
        server_module
    )
    actual = {t.name for t in tools}
    unexpected = actual - known
    assert not unexpected, (
        f"{server_module}: found unexpected tools {unexpected}. "
        f"Add them to EXPECTED_READONLY or EXPECTED_WRITE."
    )


@pytest.mark.parametrize(
    "server_module",
    sorted(LEGACY_CONTENT_ACCESS_TOOLS),
    ids=sorted(LEGACY_CONTENT_ACCESS_TOOLS),
)
def test_legacy_content_access_tools_are_not_exposed(server_module: str):
    """Content access should be exposed through read_* tool names only."""
    tools = _list_tools(server_module)
    actual = {tool.name for tool in tools}

    assert not (actual & LEGACY_CONTENT_ACCESS_TOOLS[server_module])


@pytest.mark.parametrize(
    ("server_module", "tool_name", "parameter_name", "is_required"),
    SEARCH_PARAMETER_DESCRIPTIONS,
    ids=[
        f"{server_module}.{tool_name}.{parameter_name}"
        for (
            server_module,
            tool_name,
            parameter_name,
            _is_required,
        ) in SEARCH_PARAMETER_DESCRIPTIONS
    ],
)
def test_search_parameters_warn_not_to_use_summary(
    server_module: str,
    tool_name: str,
    parameter_name: str,
    is_required: bool,
):
    """Search-like tools should make search argument placement explicit."""
    tool = _tool_by_name(server_module, tool_name)
    parameter = tool.parameters["properties"][parameter_name]
    description = parameter["description"]

    if is_required:
        assert "Required" in description
    else:
        assert "Optional" in description
    assert f"in `{parameter_name}`" in description
    assert "do not put" in description
    assert "summary" in description
