"""
客户端使用示例 - 展示如何配置 OpenAI SDK 使用代理
"""
import os
from openai import OpenAI


def example_with_url_encoded_config():
    """示例1: 使用 URL 编码的配置（适合临时测试）"""
    import json
    import urllib.parse
    
    config = {
        "basic_moderation": {"enabled": True, "keywords_file": "configs/keywords.txt"},
        "smart_moderation": {"enabled": True, "profile": "default"},
        "format_transform": {"enabled": False}
    }
    
    cfg_str = json.dumps(config, separators=(',', ':'))
    cfg_enc = urllib.parse.quote(cfg_str, safe='')
    upstream = "https://api.openai.com/v1"
    base_url = f"http://localhost:8000/{cfg_enc}${upstream}"
    
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)


def example_with_env_config():
    """示例2: 使用环境变量配置（推荐，URL更短）"""
    # 需要在 .env 中配置: PROXY_CONFIG_DEFAULT=<json配置>
    upstream = "https://api.openai.com/v1"
    base_url = f"http://localhost:8000/!PROXY_CONFIG_DEFAULT${upstream}"
    
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)


def example_openai_to_claude():
    """示例3: OpenAI 格式转换为 Claude（使用环境变量配置）"""
    # 需要在 .env 中配置: PROXY_CONFIG_CLAUDE=<json配置>
    upstream = "https://api.anthropic.com/v1"
    base_url = f"http://localhost:8000/!PROXY_CONFIG_CLAUDE${upstream}"
    
    client = OpenAI(
        api_key=os.getenv("ANTHROPIC_API_KEY"),  # 使用 Claude API Key
        base_url=base_url
    )
    
    # 使用 OpenAI SDK 格式，代理会自动转换为 Claude 格式
    response = client.chat.completions.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    print("选择示例:")
    print("1. URL 编码配置")
    print("2. 环境变量配置（推荐）")
    print("3. OpenAI -> Claude 转换")
    
    choice = input("输入选项 (1-3): ").strip()
    
    if choice == "1":
        example_with_url_encoded_config()
    elif choice == "2":
        example_with_env_config()
    elif choice == "3":
        example_openai_to_claude()
    else:
        print("无效选项")