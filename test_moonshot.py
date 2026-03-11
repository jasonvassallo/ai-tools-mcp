#!/usr/bin/env python3
"""Simple test script for Moonshot Kimi K2 Thinking API"""

import subprocess
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
            f"Could not find keychain item. Add it with:\n"
            f"  security add-generic-password -s '{service}' -a '{account}' -w 'YOUR_API_KEY'"
        )
    return result.stdout.strip()


# Get API key from macOS Keychain
API_KEY = get_api_key_from_keychain("moonshot-api", "kimi")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.moonshot.ai/v1"  # .ai not .cn
)

print("Testing Kimi K2 Thinking API...")
print("-" * 40)

response = client.chat.completions.create(
    model="kimi-k2-thinking",  # Thinking model
    messages=[
        {
            "role": "system",
            "content": "You are Kimi, an AI assistant created by Moonshot AI."
        },
        {
            "role": "user",
            "content": "What is 15 + 27?"
        }
    ],
    temperature=1.0,  # Recommended for thinking model
    max_tokens=1024
)

# Get the response
message = response.choices[0].message

# Check for reasoning content (thinking model feature)
reasoning = getattr(message, "reasoning_content", None)
if reasoning:
    print("REASONING:")
    print(reasoning)
    print("-" * 40)

print("ANSWER:")
print(message.content)

print("-" * 40)
print("Success! Kimi K2 Thinking works.")
