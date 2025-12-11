"""
格式解析器 - 支持多来源自动检测
"""
from typing import Dict, Any, Optional, Tuple, List, Protocol
from ai_proxy.transform.formats.internal_models import InternalChatRequest, InternalChatResponse
from ai_proxy.transform.formats import openai_chat, claude_chat, openai_codex, gemini_chat


class FormatParser(Protocol):
    """格式解析器接口"""
    name: str
    
    def can_parse(self, path: str, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
        """判断是否能解析该格式"""
        ...
    
    def from_format(self, body: Dict[str, Any]) -> InternalChatRequest:
        """从特定格式转为内部格式"""
        ...
    
    def to_format(self, req: InternalChatRequest) -> Dict[str, Any]:
        """从内部格式转为特定格式"""
        ...
    
    def get_target_path(self, req: InternalChatRequest, original_path: str) -> str:
        """
        获取目标格式的请求路径
        
        Args:
            req: 内部格式请求对象
            original_path: 原始请求路径
            
        Returns:
            目标格式的请求路径
        """
        ...
    
    def resp_to_internal(self, resp: Dict[str, Any]) -> InternalChatResponse:
        """响应转为内部格式"""
        ...
    
    def internal_to_resp(self, resp: InternalChatResponse) -> Dict[str, Any]:
        """内部格式转为响应"""
        ...


class OpenAIChatParser:
    """OpenAI Chat 解析器"""
    name = "openai_chat"
    
    def can_parse(self, path: str, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
        return openai_chat.can_parse_openai_chat(path, headers, body)
    
    def from_format(self, body: Dict[str, Any]) -> InternalChatRequest:
        return openai_chat.from_openai_chat(body)
    
    def to_format(self, req: InternalChatRequest) -> Dict[str, Any]:
        return openai_chat.to_openai_chat(req)
    
    def get_target_path(self, req: InternalChatRequest, original_path: str) -> str:
        """OpenAI Chat 格式的标准路径"""
        return "/v1/chat/completions"
    
    def resp_to_internal(self, resp: Dict[str, Any]) -> InternalChatResponse:
        return openai_chat.openai_chat_resp_to_internal(resp)
    
    def internal_to_resp(self, resp: InternalChatResponse) -> Dict[str, Any]:
        return openai_chat.internal_to_openai_resp(resp)


class ClaudeChatParser:
    """Claude Chat 解析器"""
    name = "claude_chat"
    
    def can_parse(self, path: str, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
        return claude_chat.can_parse_claude_chat(path, headers, body)
    
    def from_format(self, body: Dict[str, Any]) -> InternalChatRequest:
        return claude_chat.from_claude_chat(body)
    
    def to_format(self, req: InternalChatRequest) -> Dict[str, Any]:
        return claude_chat.to_claude_chat(req)
    
    def get_target_path(self, req: InternalChatRequest, original_path: str) -> str:
        """Claude Chat 格式的标准路径"""
        return "/v1/messages"
    
    def resp_to_internal(self, resp: Dict[str, Any]) -> InternalChatResponse:
        return claude_chat.claude_resp_to_internal(resp)
    
    def internal_to_resp(self, resp: InternalChatResponse) -> Dict[str, Any]:
        return claude_chat.internal_to_claude_resp(resp)


class OpenAICodexParser:
    """OpenAI Codex/Completions 解析器"""
    name = "openai_codex"
    
    def can_parse(self, path: str, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
        return openai_codex.can_parse_openai_codex(path, headers, body)
    
    def from_format(self, body: Dict[str, Any]) -> InternalChatRequest:
        return openai_codex.from_openai_codex(body)
    
    def to_format(self, req: InternalChatRequest) -> Dict[str, Any]:
        return openai_codex.to_openai_codex(req)
    
    def get_target_path(self, req: InternalChatRequest, original_path: str) -> str:
        """OpenAI Codex 格式的标准路径"""
        return "/v1/completions"
    
    def resp_to_internal(self, resp: Dict[str, Any]) -> InternalChatResponse:
        return openai_codex.openai_codex_resp_to_internal(resp)
    
    def internal_to_resp(self, resp: InternalChatResponse) -> Dict[str, Any]:
        return openai_codex.internal_to_openai_codex_resp(resp)


class GeminiChatParser:
    """Gemini Chat 解析器"""
    name = "gemini_chat"
    
    def __init__(self):
        self._last_path = ""  # 存储最后一次解析的路径
    
    def can_parse(self, path: str, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
        self._last_path = path  # 保存路径供 from_format 使用
        return gemini_chat.can_parse_gemini_chat(path, headers, body)
    
    def from_format(self, body: Dict[str, Any]) -> InternalChatRequest:
        # 使用保存的路径来判断是否流式
        return gemini_chat.from_gemini_chat(body, self._last_path)
    
    def to_format(self, req: InternalChatRequest) -> Dict[str, Any]:
        return gemini_chat.to_gemini_chat(req)
    
    def get_target_path(self, req: InternalChatRequest, original_path: str) -> str:
        """
        Gemini Chat 格式的路径（需要包含模型名）
        格式：/v1beta/models/{model}:generateContent 或 :streamGenerateContent
        """
        model = req.model or "gemini-2.5-flash"
        
        # 判断是否流式
        if req.stream:
            return f"/v1beta/models/{model}:streamGenerateContent"
        else:
            return f"/v1beta/models/{model}:generateContent"
    
    def resp_to_internal(self, resp: Dict[str, Any]) -> InternalChatResponse:
        return gemini_chat.gemini_resp_to_internal(resp)
    
    def internal_to_resp(self, resp: InternalChatResponse) -> Dict[str, Any]:
        return gemini_chat.internal_to_gemini_resp(resp)


# 注册所有解析器
# 注意：Gemini 格式必须放在最前面,因为它的特征最明显
# 这样可以防止 Gemini 格式被误识别为 OpenAI 或 Claude 格式
PARSERS: Dict[str, FormatParser] = {
    "gemini_chat": GeminiChatParser(),
    "openai_chat": OpenAIChatParser(),
    "claude_chat": ClaudeChatParser(),
    "openai_codex": OpenAICodexParser(),
}


def detect_and_parse(
    config_from: Any,
    path: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    strict_parse: bool = False,
    disable_tools: bool = False
) -> Tuple[Optional[str], Optional[InternalChatRequest], Optional[str]]:
    """
    检测并解析请求格式
    
    Args:
        config_from: 配置的来源格式，可以是：
            - str: 单一格式名称
            - list: 格式名称列表
            - "auto": 自动检测所有支持的格式
        path: 请求路径
        headers: 请求头
        body: 请求体
        strict_parse: 是否启用严格解析模式
        disable_tools: 是否禁用工具调用（如果启用，将拒绝包含工具的请求）
    
    Returns:
        (格式名称, 内部请求对象, 错误消息) 或 (None, None, 错误消息) 表示无法识别
    """
    # 1. 如果禁用工具，排除仅支持工具的格式
    if disable_tools:
        # Claude Code 和 OpenAI Codex 主要用于工具调用，应该被排除
        tool_only_formats = ["openai_codex"]
    else:
        tool_only_formats = []
    
    # 2. 确定候选格式列表
    if config_from == "auto":
        candidates = [f for f in PARSERS.keys() if f not in tool_only_formats]
    elif isinstance(config_from, str):
        if disable_tools and config_from in tool_only_formats:
            # 如果配置要求使用被禁用的格式，返回错误
            return None, None, (
                f"Format '{config_from}' is not allowed when tools are disabled. "
                f"Formats '{', '.join(tool_only_formats)}' are designed for tool calling and cannot be used with disable_tools=true. "
                f"Please use 'openai_chat' or 'claude_chat' instead."
            )
        candidates = [config_from]
    elif isinstance(config_from, list):
        candidates = [f for f in config_from if f not in tool_only_formats]
    else:
        candidates = [f for f in PARSERS.keys() if f not in tool_only_formats]
    
    # 3. 按顺序尝试解析
    for name in candidates:
        parser = PARSERS.get(name)
        if parser is None:
            continue
        
        try:
            if parser.can_parse(path, headers, body):
                internal = parser.from_format(body)
                
                # 如果禁用工具，检查请求中是否包含工具
                if disable_tools:
                    has_tools = _check_has_tools(internal)
                    if has_tools:
                        return None, None, (
                            f"Tool calling is disabled by configuration. "
                            f"The request contains tool definitions or tool-related content, which is not allowed. "
                            f"Please remove 'tools', 'tool_choice', tool calls, or tool results from your request."
                        )
                
                return name, internal, None
        except Exception as e:
            # 解析失败，继续尝试下一个
            print(f"[WARN] Failed to parse as {name}: {e}")
            continue
    
    # 4. 都不识别
    if strict_parse:
        # 严格模式：检查是否有其他格式可以解析
        all_formats = list(PARSERS.keys())
        excluded_formats = [f for f in all_formats if f not in candidates]
        
        # 遍历被排除的格式，看是否有可以解析的
        detectable_formats = []
        for name in excluded_formats:
            parser = PARSERS.get(name)
            if parser is None:
                continue
            
            try:
                if parser.can_parse(path, headers, body):
                    detectable_formats.append(name)
            except Exception:
                continue
        
        if detectable_formats:
            # 发现有可以解析但被排除的格式
            expected_str = f"'{config_from}'" if isinstance(config_from, str) else str(candidates)
            detected_str = ", ".join(f"'{f}'" for f in detectable_formats)
            error_msg = (
                f"Format mismatch: Request appears to be in format [{detected_str}], "
                f"but only [{expected_str}] is allowed. "
                f"Please check your 'from' configuration or update it to include the detected format."
            )
            return None, None, error_msg
        else:
            # 没有任何格式可以解析
            expected_str = f"'{config_from}'" if isinstance(config_from, str) else str(candidates)
            error_msg = (
                f"Unable to parse request format. Expected format: {expected_str}. "
                f"Please verify your request body structure matches the expected format."
            )
            return None, None, error_msg
    
    # 非严格模式：返回 None 表示无法识别（将透传）
    return None, None, None


def _check_has_tools(internal: InternalChatRequest) -> bool:
    """
    检查内部请求是否包含工具相关内容
    
    Returns:
        True 如果包含工具定义、工具调用或工具结果
    """
    # 检查是否有工具定义
    if internal.tools:
        return True
    
    # 检查 tool_choice
    if internal.tool_choice is not None:
        return True
    
    # 检查消息中是否包含工具调用或工具结果
    for msg in internal.messages:
        for block in msg.content:
            if block.type in ["tool_call", "tool_result"]:
                return True
    
    return False


def get_parser(format_name: str) -> Optional[FormatParser]:
    """获取指定格式的解析器"""
    return PARSERS.get(format_name)