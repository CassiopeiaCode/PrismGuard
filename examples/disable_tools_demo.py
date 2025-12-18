"""
示例：disable_tools=true 禁用工具调用

目标：
- 演示 format_transform.disable_tools=true 时，包含 tools/tool_choice/tool calls/tool results 的请求会被拒绝
- 该拒绝发生在代理侧解析阶段，因此不依赖上游服务是否支持工具调用
"""

import json
import os
import sys
import urllib.parse
from typing import Dict, Any

from openai import OpenAI


def build_base_url_with_inline_config(
    config: Dict[str, Any],
    upstream: str,
    proxy_host: str = "http://localhost:8000",
) -> str:
    cfg_str = json.dumps(config, separators=(",", ":"), ensure_ascii=False)
    cfg_enc = urllib.parse.quote(cfg_str, safe="")
    return f"{proxy_host}/{cfg_enc}${upstream}"


def main() -> int:
    proxy_host = os.getenv("PRISMGUARD_PROXY", "http://localhost:8000").rstrip("/")
    upstream = os.getenv("PRISMGUARD_UPSTREAM", "https://api.openai.com/v1").rstrip("/")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("PRISMGUARD_API_KEY")
    if not api_key:
        print(
            "Missing API key: set OPENAI_API_KEY (recommended) or PRISMGUARD_API_KEY",
            file=sys.stderr,
        )
        return 2

    config = {
        "basic_moderation": {"enabled": False},
        "smart_moderation": {"enabled": False},
        "format_transform": {
            "enabled": True,
            "from": "auto",
            "to": "openai_chat",
            "disable_tools": True,
            "strict_parse": True,
        },
    }

    base_url = build_base_url_with_inline_config(config, upstream, proxy_host=proxy_host)
    print(f"[INFO] base_url={base_url}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    try:
        client.chat.completions.create(
            model=os.getenv("PRISMGUARD_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": "北京天气怎么样？"}],
            tools=tools,
        )
        print(
            "[UNEXPECTED] request succeeded, but disable_tools=true should have blocked it",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print("[EXPECTED] blocked by proxy")
        print(f"[ERROR] {type(e).__name__}: {e}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())