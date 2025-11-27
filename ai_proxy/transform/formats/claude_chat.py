"""
Claude Chat 格式转换 - 支持工具调用
"""
import json
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


def can_parse_claude_chat(path: str, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
    """判断是否为 Claude Chat 格式"""
    # 排斥 Claude Code 格式：如果有 prompt 字段且没有 messages，则不是 Claude Chat
    if "prompt" in body and "messages" not in body:
        return False
    
    # 排斥带有 role="tool" 的 OpenAI Chat 格式
    # Claude Chat 不使用 role="tool"，而是使用 content 中的 tool_result type
    if "messages" in body:
        messages = body.get("messages", [])
        if messages and isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "tool":
                    return False
    
    # 检查路径
    if "/messages" in path:
        return True
    # 检查 header
    if "anthropic-version" in headers:
        return True
    # 检查 body 结构
    if "messages" in body:
        messages = body.get("messages", [])
        if messages and isinstance(messages, list):
            first_msg = messages[0]
            if isinstance(first_msg, dict):
                content = first_msg.get("content", [])
                if isinstance(content, list) and content:
                    if isinstance(content[0], dict) and "type" in content[0]:
                        return True
    return False


def from_claude_chat(body: Dict[str, Any]) -> InternalChatRequest:
    """
    Claude Chat 格式 -> 内部格式（支持工具调用）
    """
    # 解析工具定义
    tools = []
    for t in body.get("tools", []):
        tools.append(InternalTool(
            name=t["name"],
            description=t.get("description"),
            input_schema=t.get("input_schema", {})
        ))
    
    messages = []
    
    # 处理 system
    system = body.get("system", "")
    if system:
        messages.append(InternalMessage(
            role="system",
            content=[InternalContentBlock(type="text", text=system)]
        ))
    
    # 处理 messages
    for msg in body.get("messages", []):
        blocks = []
        content_parts = msg.get("content", [])
        
        # 处理字符串内容
        if isinstance(content_parts, str):
            blocks.append(InternalContentBlock(type="text", text=content_parts))
        else:
            # 处理数组内容
            for c in content_parts:
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
                
                elif ctype == "tool_result":
                    output = c.get("content", "")
                    # 如果 content 是列表，提取文本
                    if isinstance(output, list):
                        texts = [item.get("text", "") for item in output if item.get("type") == "text"]
                        output = "\n".join(texts)
                    
                    blocks.append(InternalContentBlock(
                        type="tool_result",
                        tool_result=InternalToolResult(
                            call_id=c.get("tool_use_id", ""),
                            name=None,
                            output=output
                        )
                    ))
        
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
                    result_content = [{"type": "text", "text": json.dumps(result_content, ensure_ascii=False)}]
                
                content.append({
                    "type": "tool_result",
                    "tool_use_id": b.tool_result.call_id,
                    "content": result_content
                })
        
        if content:
            role = "user" if m.role == "user" else "assistant"
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