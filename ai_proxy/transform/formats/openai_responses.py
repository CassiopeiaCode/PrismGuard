"""
OpenAI Responses API 格式转换
"""
import orjson
from typing import Dict, Any, List, Optional

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


SUPPORTED_EVENT_TYPES = {
    "response.created",
    "response.in_progress",
    "response.output_text.delta",
    "response.output_item.added",
    "response.function_call_arguments.delta",
    "response.function_call.delta",
    "response.reasoning_summary_text.delta",
    "response.completed",
    "response.failed",
    "response.incomplete",
    "error",
}


def can_parse_openai_responses(path: str, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
    """判断是否为 OpenAI Responses API 格式"""
    path = path or ""
    if "/responses" in path:
        return True

    if "input" in body and "model" in body:
        # Responses API 使用 input 字段
        return True

    # 响应结构
    if body.get("object") == "response" and "output" in body:
        return True

    return False


def from_openai_responses(body: Dict[str, Any]) -> InternalChatRequest:
    """OpenAI Responses -> 内部格式"""
    messages: List[InternalMessage] = []

    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append(
            InternalMessage(
                role="system",
                content=[InternalContentBlock(type="text", text=instructions.strip())],
            )
        )

    input_data = body.get("input")
    messages.extend(_parse_input_messages(input_data))

    tools = []
    for tool in body.get("tools", []):
        if tool.get("type") != "function":
            continue
        # Support both:
        # - {"type":"function","function":{"name":...,"parameters":...}}
        # - {"type":"function","name":...,"parameters":...}
        fn = tool.get("function", {}) if isinstance(tool.get("function"), dict) else {}
        name = tool.get("name") or fn.get("name") or ""
        description = tool.get("description") if "description" in tool else fn.get("description")
        parameters = tool.get("parameters") if "parameters" in tool else fn.get("parameters")
        tools.append(
            InternalTool(
                name=name,
                description=description,
                input_schema=parameters or {},
            )
        )

    extra_keys = {
        "model",
        "instructions",
        "input",
        "tools",
        "tool_choice",
        "stream",
        "response_format",
        "metadata",
        "max_output_tokens",
        "temperature",
        "top_p",
        "reasoning",
        "user",
        "parallel_tool_calls",
        "store",
        "service_tier",
    }
    extra = {k: v for k, v in body.items() if k not in extra_keys}

    if "response_format" in body and "response_format" not in extra:
        extra["response_format"] = body["response_format"]
    for key in ["metadata", "max_output_tokens", "temperature", "top_p", "reasoning", "parallel_tool_calls", "store", "service_tier", "user"]:
        if key in body and key not in extra:
            extra[key] = body[key]

    return InternalChatRequest(
        messages=messages,
        model=body.get("model", ""),
        stream=bool(body.get("stream")),
        tools=tools,
        tool_choice=body.get("tool_choice"),
        extra=extra,
    )


def to_openai_responses(req: InternalChatRequest) -> Dict[str, Any]:
    """内部格式 -> OpenAI Responses 请求体"""
    system_texts: List[str] = []
    normal_messages: List[InternalMessage] = []
    for msg in req.messages:
        if msg.role == "system":
            text = _join_text_blocks(msg.content)
            if text:
                system_texts.append(text)
        else:
            normal_messages.append(msg)

    body: Dict[str, Any] = {}
    body.update(_normalize_extra_for_openai_responses(req.extra))
    body["model"] = req.model
    body["stream"] = req.stream

    instructions = "\n\n".join(t.strip() for t in system_texts if t.strip())
    if instructions:
        body["instructions"] = instructions

    if normal_messages:
        body["input"] = [_message_to_input_item(msg) for msg in normal_messages]
    else:
        body["input"] = ""

    if req.tools:
        body["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in req.tools
        ]

    if req.tool_choice is not None:
        body["tool_choice"] = _normalize_tool_choice_for_openai_responses(req.tool_choice)

    return body


def openai_responses_resp_to_internal(resp: Dict[str, Any]) -> InternalChatResponse:
    """OpenAI Responses 响应 -> 内部格式"""
    messages: List[InternalMessage] = []
    for item in resp.get("output", []):
        converted = _output_item_to_message(item)
        if converted:
            if isinstance(converted, list):
                messages.extend(converted)
            else:
                messages.append(converted)

    if not messages:
        messages.append(
            InternalMessage(
                role="assistant",
                content=[InternalContentBlock(type="text", text="")],
            )
        )

    usage = resp.get("usage")
    if isinstance(usage, dict):
        usage = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "responses_usage": usage,
        }

    status = (resp.get("status") or "").lower()
    finish_reason = None
    if status == "completed":
        finish_reason = "stop"
    elif status == "incomplete":
        finish_reason = "length"
    elif status == "failed":
        finish_reason = "error"

    return InternalChatResponse(
        id=resp.get("id", ""),
        model=resp.get("model", ""),
        messages=messages,
        finish_reason=finish_reason,
        usage=usage,
        extra={k: v for k, v in resp.items() if k not in {"id", "object", "output", "model", "created_at", "status", "usage"}},
    )


def internal_to_openai_responses_resp(resp: InternalChatResponse) -> Dict[str, Any]:
    """内部格式 -> OpenAI Responses 响应"""
    output: List[Dict[str, Any]] = []
    for msg in resp.messages:
        output.extend(_message_to_output_items(msg))

    usage = None
    if resp.usage:
        raw_usage = resp.usage.get("responses_usage")
        if isinstance(raw_usage, dict):
            usage = raw_usage
        else:
            usage = {
                "input_tokens": resp.usage.get("prompt_tokens", 0),
                "output_tokens": resp.usage.get("completion_tokens", 0),
                "total_tokens": resp.usage.get("total_tokens", 0),
            }

    status = None
    if resp.finish_reason == "stop":
        status = "completed"
    elif resp.finish_reason == "length":
        status = "incomplete"
    elif resp.finish_reason == "error":
        status = "failed"

    response = {
        "object": "response",
        "id": resp.id,
        "model": resp.model,
        "created_at": resp.extra.get("created_at"),
        "output": output,
        "status": status,
    }
    if usage:
        response["usage"] = usage
    # Never allow extra to override required top-level fields.
    reserved = {"created_at", "object", "id", "model", "output", "status", "usage"}
    response.update({k: v for k, v in resp.extra.items() if k not in reserved})
    return response


def _parse_input_messages(input_data: Any) -> List[InternalMessage]:
    messages: List[InternalMessage] = []
    if input_data is None:
        return messages

    if isinstance(input_data, str):
        messages.append(
            InternalMessage(
                role="user",
                content=[InternalContentBlock(type="text", text=input_data)],
            )
        )
        return messages

    if isinstance(input_data, dict):
        items = [input_data]
    elif isinstance(input_data, list):
        items = input_data
    else:
        return messages

    for item in items:
        msg = _input_item_to_message(item)
        if isinstance(msg, list):
            messages.extend(msg)
        elif msg:
            messages.append(msg)

    return messages


def _input_item_to_message(item: Dict[str, Any]) -> Optional[Any]:
    if not isinstance(item, dict):
        return None

    item_type = item.get("type") or "message"
    role = item.get("role") or "user"

    if item_type in {"message", "input_text", "output_text", ""}:
        blocks = _content_to_blocks(item.get("content"), default_text=item.get("text"))
        return InternalMessage(role=role, content=blocks or [InternalContentBlock(type="text", text="")])

    if item_type in {"function_call", "tool_call"}:
        tool_call = InternalToolCall(
            id=item.get("call_id", item.get("id", "")),
            name=item.get("name", ""),
            arguments=_safe_json_loads(item.get("arguments")),
        )
        return InternalMessage(
            role="assistant",
            content=[InternalContentBlock(type="tool_call", tool_call=tool_call)],
        )

    if item_type in {"function_call_output", "tool_result"}:
        output_content = _content_to_blocks(item.get("output"), default_text=item.get("text"))
        return InternalMessage(
            role="tool",
            content=[
                InternalContentBlock(
                    type="tool_result",
                    tool_result=InternalToolResult(
                        call_id=item.get("call_id", ""),
                        name=item.get("name"),
                        output=_blocks_to_text(output_content),
                    ),
                )
            ],
        )

    if item_type == "reasoning":
        summaries = item.get("summary") or []
        text = "\n".join(s.get("text", "") for s in summaries if isinstance(s, dict))
        return InternalMessage(
            role="assistant",
            content=[InternalContentBlock(type="text", text=text)],
        )

    return None


def _content_to_blocks(content: Any, default_text: Optional[str] = None) -> List[InternalContentBlock]:
    blocks: List[InternalContentBlock] = []
    if content is None and default_text is not None:
        return [InternalContentBlock(type="text", text=default_text)]

    items: List[Dict[str, Any]] = []
    if isinstance(content, list):
        items = content
    elif isinstance(content, dict):
        if "items" in content and isinstance(content["items"], list):
            items = content["items"]
        else:
            items = [content]
    elif isinstance(content, str):
        return [InternalContentBlock(type="text", text=content)]

    for part in items:
        part_type = (part.get("type") or "").lower()
        if part_type in {"input_text", "output_text", "text"}:
            text = part.get("text")
            if text is not None:
                blocks.append(InternalContentBlock(type="text", text=text))
        elif part_type in {"image_url", "input_image"}:
            url = part.get("image_url") or part.get("image_url") or part.get("url")
            if not url and isinstance(part.get("image"), dict):
                url = part["image"].get("url")
            if url:
                blocks.append(
                    InternalContentBlock(
                        type="image_url",
                        image_url=InternalImageBlock(url=url, detail=part.get("detail")),
                    )
                )

    if not blocks and default_text is not None:
        blocks.append(InternalContentBlock(type="text", text=default_text))
    return blocks


def _output_item_to_message(item: Dict[str, Any]) -> Optional[Any]:
    if not isinstance(item, dict):
        return None

    item_type = item.get("type") or "message"
    role = item.get("role") or "assistant"

    if item_type in {"message", "output_text", "input_text", ""}:
        blocks = _content_to_blocks(item.get("content"), default_text=item.get("text"))
        return InternalMessage(role=role, content=blocks or [InternalContentBlock(type="text", text="")])

    if item_type == "reasoning":
        summaries = item.get("summary") or []
        text = "\n".join(s.get("text", "") for s in summaries if isinstance(s, dict))
        return InternalMessage(role="assistant", content=[InternalContentBlock(type="text", text=text)])

    if item_type in {"function_call", "tool_call"}:
        call = InternalToolCall(
            id=item.get("call_id", item.get("id", "")),
            name=item.get("name", ""),
            arguments=_safe_json_loads(item.get("arguments")),
        )
        return InternalMessage(
            role="assistant",
            content=[InternalContentBlock(type="tool_call", tool_call=call)],
        )

    if item_type in {"function_call_output", "tool_result"}:
        return InternalMessage(
            role="tool",
            content=[
                InternalContentBlock(
                    type="tool_result",
                    tool_result=InternalToolResult(
                        call_id=item.get("call_id", ""),
                        name=item.get("name"),
                        output=item.get("output"),
                    ),
                )
            ],
        )

    return None


def _message_to_input_item(msg: InternalMessage) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "type": "message",
        "role": msg.role,
        "content": [],
    }

    items: List[Dict[str, Any]] = []
    for block in msg.content:
        if block.type == "text" and block.text is not None:
            items.append(
                {
                    # Request-side input content uses input_* parts, regardless of message role.
                    # (Responses API allows role=user/assistant/system/developer for input messages.)
                    "type": "input_text",
                    "text": block.text,
                }
            )
        elif block.type == "image_url" and block.image_url:
            items.append(
                {
                    "type": "input_image",
                    "image_url": block.image_url.url,
                    "detail": block.image_url.detail,
                }
            )
        elif block.type == "tool_call" and block.tool_call:
            return {
                "type": "function_call",
                "id": block.tool_call.id or "",
                "call_id": block.tool_call.id,
                "name": block.tool_call.name,
                "arguments": json_dumps_text(block.tool_call.arguments),
                "status": "completed",
            }
        elif block.type == "tool_result" and block.tool_result:
            return {
                "type": "function_call_output",
                "call_id": block.tool_result.call_id,
                "name": block.tool_result.name,
                "output": block.tool_result.output,
                "status": "completed",
            }

    if not items:
        item["content"] = [{"type": "input_text", "text": ""}]
    else:
        item["content"] = items
    return item


def _message_to_output_items(msg: InternalMessage) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    text_blocks = [
        block for block in msg.content if block.type == "text" and block.text is not None
    ]
    image_blocks = [block for block in msg.content if block.type == "image_url" and block.image_url]

    if text_blocks or image_blocks:
        content_items: List[Dict[str, Any]] = []
        for block in text_blocks:
            content_items.append({"type": "output_text", "text": block.text})
        for block in image_blocks:
            content_items.append(
                {
                    "type": "input_image",
                    "image_url": block.image_url.url,
                    "detail": block.image_url.detail,
                }
            )
        items.append({"type": "message", "role": msg.role, "content": content_items})

    for block in msg.content:
        if block.type == "tool_call" and block.tool_call:
            items.append(
                {
                    "type": "function_call",
                    "call_id": block.tool_call.id,
                    "name": block.tool_call.name,
                    "arguments": json_dumps_text(block.tool_call.arguments),
                }
            )
        elif block.type == "tool_result" and block.tool_result:
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": block.tool_result.call_id,
                    "name": block.tool_result.name,
                    "output": block.tool_result.output,
                }
            )

    if not items:
        items.append(
            {"type": "message", "role": msg.role, "content": [{"type": "output_text", "text": ""}]}
        )
    return items


def _join_text_blocks(blocks: List[InternalContentBlock]) -> str:
    texts = [block.text for block in blocks if block.type == "text" and block.text]
    return "\n".join(texts)


def _blocks_to_text(blocks: List[InternalContentBlock]) -> str:
    texts = [block.text for block in blocks if block.type == "text" and block.text]
    return "\n".join(texts)


def _safe_json_loads(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            return json_loads(data)
        except Exception:
            return {}
    return {}


def _normalize_tool_choice_for_openai_responses(tool_choice: Any) -> Any:
    """
    Convert common tool_choice shapes from other APIs into Responses-compatible shapes.

    Examples:
    - ChatCompletions: {"type":"function","function":{"name":"x"}} -> {"type":"function","name":"x"}
    - Claude: {"type":"tool","name":"x"} -> {"type":"function","name":"x"}
    """
    if not isinstance(tool_choice, dict):
        return tool_choice

    t = tool_choice.get("type")
    if t == "function":
        fn = tool_choice.get("function")
        if isinstance(fn, dict) and fn.get("name") and "name" not in tool_choice:
            return {"type": "function", "name": fn.get("name")}
        if "name" in tool_choice:
            return {"type": "function", "name": tool_choice.get("name")}
        return tool_choice

    if t == "tool" and tool_choice.get("name"):
        return {"type": "function", "name": tool_choice.get("name")}

    return tool_choice


def _normalize_extra_for_openai_responses(extra: Dict[str, Any]) -> Dict[str, Any]:
    """
    ChatCompletions -> Responses param normalization.

    We do NOT attempt a strict allowlist here (to avoid breaking newer parameters),
    but we drop or rewrite a few known-incompatible fields that frequently cause 400s.
    """
    if not isinstance(extra, dict) or not extra:
        return {}

    out = dict(extra)

    # max_tokens (ChatCompletions) -> max_output_tokens (Responses)
    if "max_output_tokens" not in out:
        if isinstance(out.get("max_tokens"), int):
            out["max_output_tokens"] = out.pop("max_tokens")
        elif isinstance(out.get("max_completion_tokens"), int):
            out["max_output_tokens"] = out.pop("max_completion_tokens")

    # response_format (ChatCompletions) -> text.format (Responses)
    # ChatCompletions: {"type":"json_schema","json_schema":{"name":...,"schema":...,"strict":...}}
    # Responses:       {"type":"json_schema","name":...,"schema":...,"description":...,"strict":...}
    rf = out.pop("response_format", None)
    if rf is not None and "text" not in out:
        fmt = None
        if isinstance(rf, dict) and rf.get("type") == "json_schema":
            js = rf.get("json_schema") if isinstance(rf.get("json_schema"), dict) else {}
            fmt = {
                "type": "json_schema",
                "name": js.get("name") or "response",
                "schema": js.get("schema") or {},
            }
            if "description" in js:
                fmt["description"] = js.get("description")
            if "strict" in js:
                fmt["strict"] = js.get("strict")
        elif isinstance(rf, dict) and rf.get("type") in {"json_object", "text"}:
            fmt = {"type": rf.get("type")}

        if fmt is not None:
            out["text"] = {"format": fmt}

    # stream_options.include_usage exists on ChatCompletions; Responses has different stream_options fields.
    so = out.get("stream_options")
    if isinstance(so, dict) and ("include_usage" in so):
        so = dict(so)
        so.pop("include_usage", None)
        if so:
            out["stream_options"] = so
        else:
            out.pop("stream_options", None)

    # Drop known ChatCompletions-only keys that commonly break Responses.
    for k in [
        "messages",
        "n",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "logprobs",
    ]:
        out.pop(k, None)

    return out
