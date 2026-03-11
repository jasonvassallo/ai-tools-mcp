#!/usr/bin/env python3
"""
MCP server providing two hosted-AI tools:
1. kimi_think - Kimi K2 Thinking for extended reasoning
2. web_search - Perplexity Sonar Pro for web search with citations
"""

import subprocess
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from openai import OpenAI


def get_api_key_from_keychain(service: str, account: str) -> str:
    """Retrieve API key from macOS Keychain."""
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise ValueError(
            f"Keychain item not found. Add with:\n"
            f"  security add-generic-password -s '{service}' -a '{account}' -w 'YOUR_API_KEY'"
        )
    return result.stdout.strip()


# Initialize clients
kimi_client = OpenAI(
    api_key=get_api_key_from_keychain("moonshot-api", "kimi"),
    base_url="https://api.moonshot.ai/v1"
)

perplexity_client = OpenAI(
    api_key=get_api_key_from_keychain("perplexity-api", "sonar"),
    base_url="https://api.perplexity.ai"
)

# Create MCP server
server = Server("ai-tools-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="kimi_think",
            description=(
                "Use Kimi K2 Thinking for extended reasoning tasks. "
                "Best for: complex problem solving, multi-step reasoning, "
                "code analysis, mathematical proofs, strategic planning. "
                "Returns both the reasoning process and final answer."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The problem or question requiring deep reasoning"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens for response (default: 4096)",
                        "default": 4096
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="web_search",
            description=(
                "Use Perplexity Sonar Pro for web search with real-time data. "
                "Best for: current events, recent information, fact-checking, "
                "finding sources, research with citations. "
                "Returns answer with source citations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens for response (default: 1024)",
                        "default": 1024
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "kimi_think":
        prompt = arguments.get("prompt")
        max_tokens = arguments.get("max_tokens", 4096)

        response = kimi_client.chat.completions.create(
            model="kimi-k2-thinking",
            messages=[
                {
                    "role": "system",
                    "content": "You are Kimi, an AI assistant created by Moonshot AI."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1.0,
            max_tokens=max_tokens
        )

        message = response.choices[0].message
        reasoning = getattr(message, "reasoning_content", None)

        result = ""
        if reasoning:
            result += f"## Reasoning Process\n\n{reasoning}\n\n---\n\n"
        result += f"## Answer\n\n{message.content}"

        return [TextContent(type="text", text=result)]

    elif name == "web_search":
        query = arguments.get("query")
        max_tokens = arguments.get("max_tokens", 1024)

        response = perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful search assistant. Provide accurate, well-sourced answers."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            max_tokens=max_tokens
        )

        message = response.choices[0].message

        # Perplexity includes citations in the response
        result = f"## Search Results\n\n{message.content}"

        return [TextContent(type="text", text=result)]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
