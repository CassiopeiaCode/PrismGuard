"""
流式示例：启用 delay_stream_header 防空回复

目标：
- 演示如何在 URL 配置中开启 format_transform.delay_stream_header=true
- 演示流式请求（stream=True）在代理侧会预读内容，直到：
  - 累计文本长度 > 2，或
  - 检测到工具调用
才会放行响应头，降低“空回复/断流导致客户端误判成功”的概率。
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
            "enabled": False,
            "delay_stream_header": True,
        },
    }

    base_url = build_base_url_with_inline_config(config, upstream, proxy_host=proxy_host)
    print(f"[INFO] base_url={base_url}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    stream = client.chat.completions.create(
        model=os.getenv("PRISMGUARD_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "user", "content": "用 3 句话介绍一下 PrismGuard，并在最后加一个句号。"}
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