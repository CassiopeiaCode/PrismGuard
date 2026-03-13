"""
Claude Chat 格式转换 - 支持工具调用
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
    InternalToolResult
)


def json_loads(s: str) -> Any:
    return orjson.loads(s)


def json_dumps(obj: Any) -> bytes:
    return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS)


def json_dumps_text(obj: Any) -> str:
    return json_dumps(obj).decode("utf-8")


def can_parse_claude_chat(path: str, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
    """
    判断是否为 Claude Chat 或 Claude Code 格式
    """
    # 0. 优先排斥 Gemini Chat 格式
    # Gemini 使用 "contents" 而非 "messages"
    if "contents" in body and isinstance(body.get("contents"), list):
        contents = body.get("contents", [])
        if contents and isinstance(contents[0], dict):
            # 检查是否有 Gemini 特有的 "parts" 字段
            if "parts" in contents[0]:
                return False
    
    # 1. 优先排斥 OpenAI Chat 格式
    if "messages" in body and isinstance(body["messages"], list):
        for msg in body["messages"]:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "tool":
                return False  # 这是 OpenAI 的 role="tool"
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        # OpenAI 多模态专用格式
                        return False
    
    # 2. 检查 Claude Chat 的关键标识
    if (
        "/messages" in path
        or "anthropic-version" in {k.lower(): v for k, v in headers.items()}
        or "anthropic_version" in body
    ):
        return True
    
    # 3. 检查 Claude Chat 的 body 结构
    # Some wrappers may JSON-stringify the messages list (e.g. "messages": "[{...}]").
    if "messages" in body and isinstance(body["messages"], (list, str)):
        return True
        
    # 4. 检查 Claude Code (Agent SDK) 的 body 结构
    if "prompt" in body and isinstance(body.get("prompt"), str):
        return True
        
    return False


def _from_claude_code(body: Dict[str, Any]) -> InternalChatRequest:
    """从 Claude Code (Agent SDK) 格式转换"""
    prompt = body.get("prompt", "")
    options = body.get("options", {})
    
    messages = []
    system_prompt = options.get("systemPrompt")
    if system_prompt:
        messages.append(InternalMessage(
            role="system",
            content=[InternalContentBlock(type="text", text=system_prompt)]
        ))
    
    if isinstance(prompt, str):
        messages.append(InternalMessage(
            role="user",
            content=[InternalContentBlock(type="text", text=prompt)]
        ))
    
    tools = []
    mcp_servers = options.get("mcpServers", {})
    for server_name, server_config in mcp_servers.items():
        server_tools = server_config.get("tools", [])
        for tool_def in server_tools:
            tools.append(InternalTool(
                name=f"mcp__{server_name}__{tool_def.get('name', '')}",
                description=tool_def.get("description"),
                input_schema=tool_def.get("input_schema", {})
            ))
            
    return InternalChatRequest(
        messages=messages,
        model=options.get("model", "claude-sonnet-4-5"),
        stream=False,
        tools=tools,
        tool_choice=options.get("tool_choice"),
        extra={k: v for k, v in options.items()
               if k not in ["model", "systemPrompt", "mcpServers", "tool_choice"]}
    )

def _from_claude_chat(body: Dict[str, Any]) -> InternalChatRequest:
    """从标准 Claude Chat 格式转换"""
    tools = []
    for t in body.get("tools", []):
        tools.append(InternalTool(
            name=t["name"],
            description=t.get("description"),
            input_schema=t.get("input_schema", {})
        ))
    
    messages = []
    system_content = body.get("system")
    
    if system_content:
        system_text = ""
        if isinstance(system_content, str):
            system_text = system_content
        elif isinstance(system_content, list):
            # 处理 system 是内容块列表的情况
            texts = []
            for block in system_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            system_text = "\n".join(texts)
        
        if system_text:
            messages.append(InternalMessage(
                role="system",
                content=[InternalContentBlock(type="text", text=system_text)]
            ))
    
    raw_messages = body.get("messages", [])
    if isinstance(raw_messages, str):
        try:
            raw_messages = json_loads(raw_messages)
        except Exception:
            raw_messages = []

    for msg in raw_messages:
        blocks = []
        content_parts = msg.get("content", [])
        
        if isinstance(content_parts, str):
            blocks.append(InternalContentBlock(type="text", text=content_parts))
        else:
            # Normal Claude Messages: content is a list of blocks.
            # Some wrappers send a single block object (dict) instead of list; normalize it.
            if isinstance(content_parts, dict):
                content_parts = [content_parts]
            if not isinstance(content_parts, list):
                content_parts = []

            for c in content_parts:
                if not isinstance(c, dict):
                    continue
                ctype = c.get("type", "")
                if ctype == "text":
                    blocks.append(InternalContentBlock(type="text", text=c.get("text", "")))
                elif ctype == "thinking":
                    # Best-effort: internal model currently has no dedicated thinking blocks.
                    # Keep it parseable and allow downstream policies/converters to decide how to render it.
                    blocks.append(InternalContentBlock(type="text", text=c.get("thinking", "")))
                elif ctype == "tool_use":
                    blocks.append(
                        InternalContentBlock(
                            type="tool_call",
                            tool_call=InternalToolCall(
                                id=c.get("id", ""),
                                name=c.get("name", ""),
                                arguments=c.get("input", {}) if isinstance(c.get("input"), dict) else {},
                            ),
                        )
                    )
                elif ctype == "tool_result":
                    output = c.get("content", "")
                    if isinstance(output, list):
                        texts = [
                            item.get("text", "")
                            for item in output
                            if isinstance(item, dict) and item.get("type") == "text"
                        ]
                        output = "\n".join(texts)
                    blocks.append(
                        InternalContentBlock(
                            type="tool_result",
                            tool_result=InternalToolResult(
                                call_id=c.get("tool_use_id", ""),
                                name=None,
                                output=output,
                            ),
                        )
                    )
        
        if not blocks:
            blocks.append(InternalContentBlock(type="text", text=""))
        
        role = "user" if msg.get("role") == "user" else "assistant"
        messages.append(InternalMessage(role=role, content=blocks))
    
    return InternalChatRequest(
        messages=messages,
        model=body.get("model", ""),
        stream=body.get("stream", False),
        tools=tools,
        tool_choice=body.get("tool_choice"),
        extra={k: v for k, v in body.items()
               if k not in ["system", "messages", "model", "stream", "tools", "tool_choice"]}
    )

def from_claude_chat(body: Dict[str, Any]) -> InternalChatRequest:
    """
    Claude Chat / Code 格式 -> 内部格式
    """
    if "prompt" in body and "messages" not in body:
        return _from_claude_code(body)
    else:
        return _from_claude_chat(body)


def to_claude_chat(req: InternalChatRequest) -> Dict[str, Any]:
    """
    内部格式 -> Claude Chat 格式（支持工具调用）
    """
    # 转换工具定义
    tools = []
    for t in req.tools:
        tools.append({
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema
        })
    
    # 分离 system 和其他消息
    system_msgs = [m for m in req.messages if m.role == "system"]
    other_msgs = [m for m in req.messages if m.role != "system"]
    
    body = {
        "model": req.model,
        "stream": req.stream
    }
    
    # 添加 system
    if system_msgs:
        system_texts = []
        for m in system_msgs:
            texts = [b.text for b in m.content if b.type == "text" and b.text]
            system_texts.extend(texts)
        if system_texts:
            body["system"] = "\n".join(system_texts)
    
    # 转换消息
    claude_msgs = []
    for m in other_msgs:
        content = []
        
        for b in m.content:
            if b.type == "text" and b.text:
                content.append({"type": "text", "text": b.text})
            
            elif b.type == "tool_call" and b.tool_call:
                content.append({
                    "type": "tool_use",
                    "id": b.tool_call.id,
                    "name": b.tool_call.name,
                    "input": b.tool_call.arguments
                })
            
            elif b.type == "tool_result" and b.tool_result:
                result_content = b.tool_result.output
                if isinstance(result_content, str):
                    result_content = [{"type": "text", "text": result_content}]
                elif isinstance(result_content, dict):
                    result_content = [{"type": "text", "text": json_dumps_text(result_content)}]
                
                content.append({
                    "type": "tool_result",
                    "tool_use_id": b.tool_result.call_id,
                    "content": result_content
                })
        
        if content:
            # Claude tool_result blocks must be sent as part of a "user" message.
            # Internally, tool results may appear under role="tool" (e.g. when converting from OpenAI Chat).
            role = "user" if m.role in {"user", "tool"} else "assistant"
            claude_msgs.append({"role": role, "content": content})
    
    body["messages"] = claude_msgs
    
    if tools:
        body["tools"] = tools
    if req.tool_choice is not None:
        body["tool_choice"] = req.tool_choice
    
    body.update(req.extra)
    
    return body


def claude_resp_to_internal(resp: Dict[str, Any]) -> InternalChatResponse:
    """
    Claude 响应 -> 内部格式
    """
    blocks = []
    
    for c in resp.get("content", []):
        ctype = c.get("type", "")
        
        if ctype == "text":
            blocks.append(InternalContentBlock(type="text", text=c.get("text", "")))
        
        elif ctype == "tool_use":
            blocks.append(InternalContentBlock(
                type="tool_call",
                tool_call=InternalToolCall(
                    id=c.get("id", ""),
                    name=c.get("name", ""),
                    arguments=c.get("input", {})
                )
            ))
    
    if not blocks:
        blocks.append(InternalContentBlock(type="text", text=""))
    
    return InternalChatResponse(
        id=resp.get("id", ""),
        model=resp.get("model", ""),
        messages=[InternalMessage(role="assistant", content=blocks)],
        finish_reason=resp.get("stop_reason"),
        usage=resp.get("usage"),
        extra={k: v for k, v in resp.items() 
               if k not in ["id", "model", "content", "stop_reason", "usage"]}
    )


def internal_to_claude_resp(resp: InternalChatResponse) -> Dict[str, Any]:
    """
    内部格式 -> Claude 响应
    """
    last_msg = resp.messages[-1] if resp.messages else InternalMessage(
        role="assistant",
        content=[InternalContentBlock(type="text", text="")]
    )
    
    content = []
    for b in last_msg.content:
        if b.type == "text" and b.text:
            content.append({"type": "text", "text": b.text})
        
        elif b.type == "tool_call" and b.tool_call:
            content.append({
                "type": "tool_use",
                "id": b.tool_call.id,
                "name": b.tool_call.name,
                "input": b.tool_call.arguments
            })
    
    if not content:
        content = [{"type": "text", "text": ""}]
    
    return {
        "id": resp.id,
        "model": resp.model,
        "type": "message",
        "role": "assistant",
        "content": content,
        "stop_reason": resp.finish_reason,
        "usage": resp.usage,
        **resp.extra
    }
