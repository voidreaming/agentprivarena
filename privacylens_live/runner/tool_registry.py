"""Service/tool role registry for PrivacyLens-Live execution traces."""

from __future__ import annotations

from dataclasses import dataclass


def _prefix_tools(service_name: str, tools: frozenset[str]) -> frozenset[str]:
    return frozenset(f"{service_name}_{tool}" for tool in tools)


@dataclass(frozen=True)
class ServiceToolSpec:
    """Static tool-role metadata for one MCP service."""

    service_name: str
    server_module: str
    read_tools: frozenset[str]
    write_tools: frozenset[str] = frozenset()
    final_action_tools: frozenset[str] = frozenset()
    multimodal_tools: frozenset[str] = frozenset()
    supports_forced_read: bool = False

    @property
    def prefixed_read_tools(self) -> frozenset[str]:
        return _prefix_tools(self.service_name, self.read_tools)

    @property
    def prefixed_write_tools(self) -> frozenset[str]:
        return _prefix_tools(self.service_name, self.write_tools)

    @property
    def prefixed_final_action_tools(self) -> frozenset[str]:
        return _prefix_tools(self.service_name, self.final_action_tools)

    @property
    def prefixed_multimodal_tools(self) -> frozenset[str]:
        return _prefix_tools(self.service_name, self.multimodal_tools)


SERVICE_TOOL_SPECS: dict[str, ServiceToolSpec] = {
    "bookstack": ServiceToolSpec(
        service_name="bookstack",
        server_module="bookstack_server",
        read_tools=frozenset({"search_pages", "read_page", "list_pages"}),
        write_tools=frozenset({"create_page", "update_page", "delete_page"}),
        final_action_tools=frozenset({"create_page"}),
        supports_forced_read=True,
    ),
    "mattermost": ServiceToolSpec(
        service_name="mattermost",
        server_module="mattermost_server",
        read_tools=frozenset({"read_messages", "search_messages", "list_users"}),
        write_tools=frozenset({"send_message"}),
        final_action_tools=frozenset({"send_message"}),
        supports_forced_read=True,
    ),
    "rocketchat": ServiceToolSpec(
        service_name="rocketchat",
        server_module="rocketchat_server",
        read_tools=frozenset(
            {
                "search_messages",
                "list_channels",
                "read_channel_history",
                "search_users",
                "get_user_info",
            }
        ),
        write_tools=frozenset({"send_channel_message"}),
        final_action_tools=frozenset({"send_channel_message"}),
        supports_forced_read=True,
    ),
    "mailpit": ServiceToolSpec(
        service_name="mailpit",
        server_module="mailpit_server",
        read_tools=frozenset({"search_emails", "read_email", "list_contacts"}),
        write_tools=frozenset({"send_email"}),
        final_action_tools=frozenset({"send_email"}),
        supports_forced_read=True,
    ),
    "gotosocial": ServiceToolSpec(
        service_name="gotosocial",
        server_module="gotosocial_server",
        read_tools=frozenset(
            {
                "read_profile",
                "read_user_posts",
                "search_users",
                "search_posts",
            }
        ),
        write_tools=frozenset({"create_post"}),
        final_action_tools=frozenset({"create_post"}),
        supports_forced_read=True,
    ),
    "radicale": ServiceToolSpec(
        service_name="radicale",
        server_module="radicale_server",
        read_tools=frozenset({"search_events", "read_event", "list_events"}),
        supports_forced_read=True,
    ),
    "google_drive": ServiceToolSpec(
        service_name="google_drive",
        server_module="google_drive_server",
        read_tools=frozenset(
            {
                "search_files",
                "get_file_metadata",
                "get_file_image",
                "describe_image",
            }
        ),
        multimodal_tools=frozenset({"get_file_image", "describe_image"}),
        supports_forced_read=True,
    ),
}

SERVICE_BY_SERVER_MODULE: dict[str, ServiceToolSpec] = {
    spec.server_module: spec for spec in SERVICE_TOOL_SPECS.values()
}

# Bare final-action tool names. Kept for evaluator compatibility.
SEND_ACTIONS = frozenset(
    tool for spec in SERVICE_TOOL_SPECS.values() for tool in spec.final_action_tools
)

# Prefixed final-action tool names. The SDK auto-prefixes MCP tool names with
# the server name, so both bare and prefixed forms are recognized.
SEND_PREFIXED = frozenset(
    tool
    for spec in SERVICE_TOOL_SPECS.values()
    for tool in spec.prefixed_final_action_tools
)

WRITE_TOOLS = frozenset(
    tool for spec in SERVICE_TOOL_SPECS.values() for tool in spec.write_tools
) | frozenset(
    tool for spec in SERVICE_TOOL_SPECS.values() for tool in spec.prefixed_write_tools
)

READ_TOOLS = frozenset(
    tool for spec in SERVICE_TOOL_SPECS.values() for tool in spec.read_tools
) | frozenset(
    tool for spec in SERVICE_TOOL_SPECS.values() for tool in spec.prefixed_read_tools
)

MULTIMODAL_TOOLS = frozenset(
    tool for spec in SERVICE_TOOL_SPECS.values() for tool in spec.multimodal_tools
) | frozenset(
    tool
    for spec in SERVICE_TOOL_SPECS.values()
    for tool in spec.prefixed_multimodal_tools
)

FINAL_ACTION_TOOLS = SEND_ACTIONS | SEND_PREFIXED
NON_READ_TOOLS = WRITE_TOOLS | {"finish", "think"}


def known_service_names() -> tuple[str, ...]:
    return tuple(SERVICE_TOOL_SPECS)


def service_spec(service_name: str) -> ServiceToolSpec:
    return SERVICE_TOOL_SPECS[service_name]


def server_module_spec(server_module: str) -> ServiceToolSpec:
    return SERVICE_BY_SERVER_MODULE[server_module]


def read_tools_for_server_module(server_module: str) -> frozenset[str]:
    return server_module_spec(server_module).read_tools


def write_tools_for_server_module(server_module: str) -> frozenset[str]:
    return server_module_spec(server_module).write_tools


def is_known_service(service_name: str) -> bool:
    return service_name in SERVICE_TOOL_SPECS


def is_read_tool(tool_name: str) -> bool:
    return tool_name in READ_TOOLS


def is_write_tool(tool_name: str) -> bool:
    return tool_name in WRITE_TOOLS


def is_multimodal_tool(tool_name: str) -> bool:
    return tool_name in MULTIMODAL_TOOLS


def is_final_action_tool(tool_name: str) -> bool:
    return tool_name in FINAL_ACTION_TOOLS


def is_agent_read_tool(tool_name: str, *, forced_read: bool) -> bool:
    return not forced_read and tool_name not in NON_READ_TOOLS
