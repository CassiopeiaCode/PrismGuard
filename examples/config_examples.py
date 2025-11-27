"""
配置示例 - 展示各种使用场景
"""
import json
import urllib.parse


def create_proxy_url(config: dict, upstream: str, proxy_host: str = "http://localhost:8000") -> str:
    """创建代理 URL（URL编码方式）"""
    cfg_str = json.dumps(config, separators=(',', ':'))
    cfg_enc = urllib.parse.quote(cfg_str, safe='')
    return f"{proxy_host}/{cfg_enc}${upstream}"


def create_proxy_url_with_env(env_key: str, upstream: str, proxy_host: str = "http://localhost:8000") -> str:
    """创建代理 URL（环境变量方式，URL更短）"""
    return f"{proxy_host}/!{env_key}${upstream}"


# 示例 1: 仅基础审核
config_basic_only = {
    "basic_moderation": {
        "enabled": True,
        "keywords_file": "configs/keywords.txt"
    },
    "smart_moderation": {
        "enabled": False
    },
    "format_transform": {
        "enabled": False
    }
}

# 示例 2: 基础 + 智能审核
config_with_smart = {
    "basic_moderation": {
        "enabled": True,
        "keywords_file": "configs/keywords.txt"
    },
    "smart_moderation": {
        "enabled": True,
        "profile": "default"
    },
    "format_transform": {
        "enabled": False
    }
}

# 示例 3: OpenAI -> Claude 转换（支持工具调用）
config_openai_to_claude = {
    "basic_moderation": {
        "enabled": False
    },
    "smart_moderation": {
        "enabled": False
    },
    "format_transform": {
        "enabled": True,
        "from": "openai_chat",
        "to": "claude_chat",
        "stream": "auto"
    }
}

# 示例 4: 多来源自动检测
config_auto_detect = {
    "basic_moderation": {
        "enabled": True,
        "keywords_file": "configs/keywords.txt"
    },
    "smart_moderation": {
        "enabled": True,
        "profile": "default"
    },
    "format_transform": {
        "enabled": True,
        "from": "auto",  # 自动检测所有支持的格式
        "to": "openai_chat",
        "stream": "auto"
    }
}

# 示例 5: 指定多个来源格式
config_multi_source = {
    "basic_moderation": {
        "enabled": True,
        "keywords_file": "configs/keywords.txt"
    },
    "smart_moderation": {
        "enabled": False
    },
    "format_transform": {
        "enabled": True,
        "from": ["openai_chat", "claude_chat"],  # 只支持这两种
        "to": "openai_chat",
        "stream": "auto"
    }
}

# 示例 6: 完整配置（所有功能开启）
config_full = {
    "basic_moderation": {
        "enabled": True,
        "keywords_file": "configs/keywords.txt",
        "error_code": "BASIC_MODERATION_BLOCKED"
    },
    "smart_moderation": {
        "enabled": True,
        "profile": "default"
    },
    "format_transform": {
        "enabled": True,
        "from": "auto",
        "to": "openai_chat",
        "stream": "auto",
        "detect": {
            "by_path": True,
            "by_header": True,
            "by_body": True
        }
    }
}


if __name__ == "__main__":
    # 生成示例 URL
    print("=" * 60)
    print("代理 URL 示例（URL编码方式）")
    print("=" * 60)
    
    print("\n1. 仅基础审核:")
    url1 = create_proxy_url(config_basic_only, "https://api.openai.com/v1")
    print(f"   {url1[:100]}...")
    print(f"   长度: {len(url1)} 字符")
    
    print("\n2. 基础 + 智能审核:")
    url2 = create_proxy_url(config_with_smart, "https://api.openai.com/v1")
    print(f"   {url2[:100]}...")
    print(f"   长度: {len(url2)} 字符")
    
    print("\n3. OpenAI -> Claude 转换:")
    url3 = create_proxy_url(config_openai_to_claude, "https://api.anthropic.com/v1")
    print(f"   {url3[:100]}...")
    print(f"   长度: {len(url3)} 字符")
    
    print("\n4. 自动检测格式:")
    url4 = create_proxy_url(config_auto_detect, "https://api.openai.com/v1")
    print(f"   {url4[:100]}...")
    print(f"   长度: {len(url4)} 字符")
    
    print("\n5. 完整配置:")
    url5 = create_proxy_url(config_full, "https://api.openai.com/v1")
    print(f"   {url5[:100]}...")
    print(f"   长度: {len(url5)} 字符")
    
    print("\n" + "=" * 60)
    print("代理 URL 示例（环境变量方式 - URL更短）")
    print("=" * 60)
    print("\n需要在 .env 文件中配置:")
    print("  PROXY_CONFIG_DEFAULT=<json配置>")
    print("  PROXY_CONFIG_CLAUDE=<json配置>")
    
    print("\n1. 使用 PROXY_CONFIG_DEFAULT:")
    url_env1 = create_proxy_url_with_env("PROXY_CONFIG_DEFAULT", "https://api.openai.com/v1")
    print(f"   {url_env1}")
    print(f"   长度: {len(url_env1)} 字符")
    
    print("\n2. 使用 PROXY_CONFIG_CLAUDE:")
    url_env2 = create_proxy_url_with_env("PROXY_CONFIG_CLAUDE", "https://api.anthropic.com/v1")
    print(f"   {url_env2}")
    print(f"   长度: {len(url_env2)} 字符")
    
    print("\n对比: 环境变量方式可节省 ~200+ 字符，避免数据库字段溢出")