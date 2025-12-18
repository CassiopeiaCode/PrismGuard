"""
示例：openai_chat ↔ openai_responses 的流式互转

背景：
- PrismGuard 内置了 OpenAI Chat SSE 与 OpenAI Responses SSE 之间的流式转换器。
- 这让你可以用同一套客户端（例如 OpenAI SDK 的 chat.completions）接入不同上游能力。

用法（示例）：

1) Chat -> Responses（把客户端的 Chat 请求转换为 Responses 上游请求）：
   PowerShell:
     $env:OPENAI_API_KEY="sk-xxx"
     $env:PRISMGUARD_PROXY="http://localhost:8000"
     $env:PRISMGUARD_UPSTREAM="https://api.openai.com/v1"
     $env:MODE="chat_to_responses"
     python examples/responses_stream_transform.py

2) Responses -> Chat（上游以 Responses SSE 流式返回，代理转回 Chat chunk 给客户端消费）：
   PowerShell:
     $env:OPENAI_API_KEY="sk-xxx"
     $env:PRISMGUARD_PROXY="http://localhost:8000"
     $env:PRISMGUARD_UPSTREAM="https://api.openai.com/v1"
     $env:MODE="responses_to_chat"
     python examples/responses_stream_transform.py

说明：
- 该示例仍使用 OpenAI SDK 的 chat.completions.stream 方式消费输出。
- 是否真正命中转换，取决于你的上游路径与配置（尤其 format_transform.from/to）。
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
    mode = (os.getenv("MODE") or "responses_to_chat").strip().lower()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("PRISMGUARD_API_KEY")
    if not api_key:
        print(
            "Missing API key: set OPENAI_API_KEY (recommended) or PRISMGUARD_API_KEY",
            file=sys.stderr,
        )
        return 2

    if mode not in {"chat_to_responses", "responses_to_chat"}:
        print("MODE must be 'chat_to_responses' or 'responses_to_chat'", file=sys.stderr)
        return 2

    if mode == "chat_to_responses":
        from_format = "openai_chat"
        to_format = "openai_responses"
    else:
        from_format = "openai_responses"
        to_format = "openai_chat"

    config = {
        "basic_moderation": {"enabled": False},
        "smart_moderation": {"enabled": False},
        "format_transform": {
            "enabled": True,
            "from": from_format,
            "to": to_format,
            "strict_parse": True,
            "delay_stream_header": True,
        },
    }

    base_url = build_base_url_with_inline_config(config, upstream, proxy_host=proxy_host)
    print(f"[INFO] mode={mode}")
    print(f"[INFO] base_url={base_url}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    stream = client.chat.completions.create(
        model=os.getenv("PRISMGUARD_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "user", "content": "输出 1 句话，包含 'responses' 这个单词。"}
        ],
        stream=True,
    )

    print("[INFO] streaming output:")
    for event in stream:
        delta = event.choices[0].delta
        if delta and delta.content:
            sys.stdout.write(delta.content)
            sys.stdout.flush()

    print("\n[INFO] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())