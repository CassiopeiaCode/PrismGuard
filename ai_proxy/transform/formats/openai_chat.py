"""
OpenAI Chat 格式转换 - 支持工具调用
"""
import orjson
from typing import Dict, Any
from ai_proxy.transform.formats.internal_models import (
    InternalChatRequest,
    InternalChatResponse,
    InternalMessage,
    InternalContentBlock,
    InternalTool,
    InternalToolCall,
    InternalToolResult,
    InternalImageBlock,
)


def json_loads(s: str) -> Any:
    return orjson.loads(s)


def json_dumps(obj: Any) -> bytes:
    return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS)


def json_dumps_text(obj: Any) -> str:
    return json_dumps(obj).decode("utf-8")


def can_parse_openai_chat(path: str, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
    """判断是否为 OpenAI Chat 格式"""
    # 优先排斥 Gemini Chat 格式
    # Gemini 使用 "contents" 而非 "messages"
    if "contents" in body and isinstance(body.get("contents"), list):
        contents = body.get("contents", [])
        if contents and isinstance(contents[0], dict):
            # 检查是否有 Gemini 特有的 "parts" 字段
            if "parts" in contents[0]:
                return False
    
    # 排斥 OpenAI Codex/Completions 格式：如果有 prompt 字段且路径不含 /chat，则不是 Chat
    if "prompt" in body and "messages" not in body:
        return False
    
    # 排斥带有 cache_control 的 Claude Chat 格式
    if "messages" in body:
        messages = body.get("messages", [])
        if messages and isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content", [])
                if isinstance(content, list):
                    # 检查是否有 cache_control 字段（Claude Chat Prompt Caching 特性）
                    has_cache_control = any(
                        isinstance(block, dict) and "cache_control" in block
                        for block in content
                    )
                    if has_cache_control:
                        return False

    # 排斥 Claude Messages/Claude Code 风格：
    # - 顶层 system 字段（Claude Messages 使用独立 system，而 OpenAI Chat 没有该字段）
    # - thinking 字段（Claude extended thinking）
    # - tools 里出现 input_schema（Anthropic tools 定义）
    if "system" in body:
        return False
    if isinstance(body.get("thinking"), dict):
        return False
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        if any(isinstance(t, dict) and "input_schema" in t for t in tools):
            return False
    if "anthropic_version" in body:
        return False
    
    # 检查路径
    if "/chat/completions" in path:
        return True
    # 检查 body 结构
    if "messages" in body:
        messages = body.get("messages", [])
        if messages and isinstance(messages, list):
            first_msg = messages[0]
            if isinstance(first_msg, dict) and "role" in first_msg:
                return True
    return False


def from_openai_chat(body: Dict[str, Any]) -> InternalChatRequest:
    """
    OpenAI Chat 格式 -> 内部格式（支持工具调用）
    """
    # 解析工具定义
    tools = []
    for t in body.get("tools", []):
        if t.get("type") == "function":
            func = t["function"]
            tools.append(InternalTool(
                name=func["name"],
                description=func.get("description"),
                input_schema=func.get("parameters", {})
            ))
    
    # 解析消息
    messages = []
    for msg in body.get("messages", []):
        blocks = []
        
        # 1. 处理文本内容
        content = msg.get("content")
        if isinstance(content, str) and content:
            blocks.append(InternalContentBlock(type="text", text=content))
        elif isinstance(content, list):
            # 多部分内容，逐块解析，支持文本与图片
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    blocks.append(
                        InternalContentBlock(
                            type="text",
                            text=part.get("text") or "",
                        )
                    )
                elif part_type == "image_url":
                    image_data = part.get("image_url") or {}
                    url = image_data.get("url")
                    if not url:
                        continue
                    blocks.append(
                        InternalContentBlock(
                            type="image_url",
                            image_url=InternalImageBlock(
                                url=url,
                                detail=image_data.get("detail"),
                            ),
                        )
                    )
        
        # 2. 处理 tool role 的消息（工具结果）
        if msg.get("role") == "tool":
            blocks.append(InternalContentBlock(
                type="tool_result",
                tool_result=InternalToolResult(
                    call_id=msg.get("tool_call_id", ""),
                    name=msg.get("name"),
                    output=msg.get("content", "")
                )
            ))
        
        # 3. 处理 assistant 的工具调用
        for tc in msg.get("tool_calls", []):
            args_str = tc.get("function", {}).get("arguments", "{}")
            try:
                args = json_loads(args_str) if isinstance(args_str, str) else args_str
            except:
                args = {}
            
            blocks.append(InternalContentBlock(
                type="tool_call",
                tool_call=InternalToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("function", {}).get("name", ""),
                    arguments=args
                )
            ))
        
        # 如果没有任何块，添加空文本块
        if not blocks:
            blocks.append(InternalContentBlock(type="text", text=""))
        
        messages.append(InternalMessage(
            role=msg.get("role", "user"),
            content=blocks
        ))
    
    return InternalChatRequest(
        messages=messages,
        model=body.get("model", ""),
        stream=body.get("stream", False),
        tools=tools,
        tool_choice=body.get("tool_choice"),
        extra={k: v for k, v in body.items() 
               if k not in ["messages", "model", "stream", "tools", "tool_choice"]}
    )


def to_openai_chat(req: InternalChatRequest) -> Dict[str, Any]:
    """
    内部格式 -> OpenAI Chat 格式（支持工具调用）
    """
    # 转换工具定义
    tools = []
    for t in req.tools:
        tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema
            }
        })
    
    # 转换消息
    messages = []
    for m in req.messages:
        # 收集不同类型的内容块
        text_blocks = [b.text for b in m.content if b.type == "text" and b.text is not None]
        tool_call_blocks = [b.tool_call for b in m.content if b.type == "tool_call"]
        tool_result_blocks = [b.tool_result for b in m.content if b.type == "tool_result"]
        image_blocks = [b.image_url for b in m.content if b.type == "image_url" and b.image_url]
        
        # 非 tool role 的消息
        if m.role != "tool":
            msg = {"role": m.role}
            
            # 添加内容块（文本/图片）
            if image_blocks:
                parts = []
                for block in m.content:
                    if block.type == "text" and block.text is not None:
                        parts.append({"type": "text", "text": block.text})
                    elif block.type == "image_url" and block.image_url:
                        image_part = {
                            "type": "image_url",
                            "image_url": {"url": block.image_url.url},
                        }
                        if block.image_url.detail:
                            image_part["image_url"]["detail"] = block.image_url.detail
                        parts.append(image_part)
                msg["content"] = parts or ""
            else:
                if text_blocks:
                    msg["content"] = "\n".join(text_blocks)
                elif not tool_call_blocks:
                    msg["content"] = ""
            
            # 添加工具调用
            if tool_call_blocks:
                msg["tool_calls"] = [{
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json_dumps_text(tc.arguments)
                    }
                } for tc in tool_call_blocks]
            
            messages.append(msg)
        
        # 工具结果转为独立的 tool 消息
        for tr in tool_result_blocks:
            messages.append({
                "role": "tool",
                "tool_call_id": tr.call_id,
                "name": tr.name,
                "content": json_dumps_text(tr.output) if isinstance(tr.output, dict) else str(tr.output)
            })
    
    # 构建请求体
    body = {
        "model": req.model,
        "messages": messages,
        "stream": req.stream
    }
    
    if tools:
        body["tools"] = tools
    if req.tool_choice is not None:
        body["tool_choice"] = req.tool_choice
    
    body.update(req.extra)
    
    return body


def openai_chat_resp_to_internal(resp: Dict[str, Any]) -> InternalChatResponse:
    """
    OpenAI Chat 响应 -> 内部格式
    """
    choice = resp.get("choices", [{}])[0]
    message = choice.get("message") or {}
    
    # 解析消息内容
    blocks = []
    
    # 文本内容
    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append(InternalContentBlock(type="text", text=content))
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                blocks.append(
                    InternalContentBlock(
                        type="text",
                        text=part.get("text") or "",
                    )
                )
            elif part_type == "image_url":
                image_data = part.get("image_url") or {}
                url = image_data.get("url")
                if not url:
                    continue
                blocks.append(
                    InternalContentBlock(
                        type="image_url",
                        image_url=InternalImageBlock(
                            url=url,
                            detail=image_data.get("detail"),
                        ),
                    )
                )
    
    # 工具调用
    for tc in message.get("tool_calls", []):
        args_str = tc.get("function", {}).get("arguments", "{}")
        try:
            args = json_loads(args_str) if isinstance(args_str, str) else args_str
        except:
            args = {}
        
        blocks.append(InternalContentBlock(
            type="tool_call",
            tool_call=InternalToolCall(
                id=tc.get("id", ""),
                name=tc.get("function", {}).get("name", ""),
                arguments=args
            )
        ))
    
    if not blocks:
        blocks.append(InternalContentBlock(type="text", text=""))
    
    return InternalChatResponse(
        id=resp.get("id", ""),
        model=resp.get("model", ""),
        messages=[InternalMessage(role="assistant", content=blocks)],
        finish_reason=choice.get("finish_reason"),
        usage=resp.get("usage"),
        extra={k: v for k, v in resp.items() 
               if k not in ["id", "model", "choices", "usage"]}
    )


def internal_to_openai_resp(resp: InternalChatResponse) -> Dict[str, Any]:
    """
    内部格式 -> OpenAI Chat 响应
    """
    # 取最后一条 assistant 消息
    last_msg = resp.messages[-1] if resp.messages else InternalMessage(
        role="assistant",
        content=[InternalContentBlock(type="text", text="")]
    )
    
    # 构建消息
    message = {"role": "assistant"}
    
    text_blocks = [b.text for b in last_msg.content if b.type == "text" and b.text is not None]
    image_blocks = [b.image_url for b in last_msg.content if b.type == "image_url" and b.image_url]

    if image_blocks:
        parts = []
        for block in last_msg.content:
            if block.type == "text" and block.text is not None:
                parts.append({"type": "text", "text": block.text})
            elif block.type == "image_url" and block.image_url:
                image_part = {
                    "type": "image_url",
                    "image_url": {"url": block.image_url.url},
                }
                if block.image_url.detail:
                    image_part["image_url"]["detail"] = block.image_url.detail
                parts.append(image_part)
        if parts:
            message["content"] = parts
    elif text_blocks:
        message["content"] = "\n".join(text_blocks)
    
    tool_calls = []
    for b in last_msg.content:
        if b.type == "tool_call" and b.tool_call:
            tool_calls.append({
                "id": b.tool_call.id,
                "type": "function",
                "function": {
                    "name": b.tool_call.name,
                    "arguments": json_dumps_text(b.tool_call.arguments)
                }
            })
    
    if tool_calls:
        message["tool_calls"] = tool_calls
    
    return {
        "id": resp.id,
        "model": resp.model,
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": resp.finish_reason
        }],
        "usage": resp.usage,
        **resp.extra
    }
