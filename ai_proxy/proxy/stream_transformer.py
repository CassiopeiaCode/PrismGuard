"""
流式响应转换器 - 在不同协议的 SSE 之间互转

支持的格式：
- openai_chat (Chat Completions SSE)
- openai_responses (Responses SSE)
- claude_chat (Anthropic Messages SSE)
- gemini_chat (Gemini alt=sse)
"""

from __future__ import annotations

import orjson
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple


def json_loads(s: Any) -> Any:
    return orjson.loads(s)


def json_dumps(obj: Any) -> bytes:
    return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS)


def _encode_sse(data: str | bytes, event: Optional[str] = None) -> bytes:
    if isinstance(data, bytes):
        if event:
            return (
                b"event: "
                + event.encode("utf-8")
                + b"\n"
                + b"data: "
                + data
                + b"\n\n"
            )
        return b"data: " + data + b"\n\n"
    if event:
        return f"event: {event}\ndata: {data}\n\n".encode("utf-8")
    return f"data: {data}\n\n".encode("utf-8")


def _encode_json(payload: dict, event: Optional[str] = None) -> bytes:
    return _encode_sse(json_dumps(payload), event=event)


def _json_dumps_compact(obj: Any) -> str:
    return json_dumps(obj).decode("utf-8")


def _safe_json_loads(s: Any) -> Any:
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        return {}
    try:
        return json_loads(s)
    except Exception:
        return {}


class SSEFrame:
    def __init__(self, event: Optional[str], data: str) -> None:
        self.event = event
        self.data = data


class BaseSSEAdapter:
    def handle_frame(self, frame: SSEFrame) -> List[bytes]:
        data = (frame.data or "").strip()
        if not data:
            return []
        if data == "[DONE]":
            return self.on_done()

        try:
            obj = json_loads(data)
        except orjson.JSONDecodeError:
            return []

        event_name = frame.event or obj.get("type") or obj.get("event")
        return self.handle_event(event_name, obj)

    def handle_event(self, event_name: Optional[str], event: dict) -> List[bytes]:
        raise NotImplementedError

    def on_done(self) -> List[bytes]:
        return [_encode_sse("[DONE]")]

    def flush(self) -> List[bytes]:
        return []


class SSEBufferTransformer:
    """SSE 分块解析器，将数据映射到指定适配器"""

    def __init__(self, adapter: BaseSSEAdapter):
        self.adapter = adapter
        self.buffer = ""

    def feed(self, chunk: bytes) -> List[bytes]:
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError:
            text = chunk.decode("utf-8", errors="ignore")

        self.buffer += text
        outputs: List[bytes] = []

        while "\n\n" in self.buffer:
            raw_event, self.buffer = self.buffer.split("\n\n", 1)
            frame = self._parse_frame(raw_event)
            if frame is None:
                continue
            outputs.extend(self.adapter.handle_frame(frame))

        return outputs

    def flush(self) -> List[bytes]:
        remaining = self.buffer.strip()
        self.buffer = ""
        outputs: List[bytes] = []
        if remaining:
            frame = self._parse_frame(remaining)
            if frame is not None:
                outputs.extend(self.adapter.handle_frame(frame))
        outputs.extend(self.adapter.flush())
        return outputs

    @staticmethod
    def _parse_frame(raw_event: str) -> Optional[SSEFrame]:
        event_name: Optional[str] = None
        data_lines: List[str] = []
        for line in raw_event.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip() or None
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
        if not data_lines and event_name is None:
            return None
        return SSEFrame(event=event_name, data="\n".join(data_lines))


class _InternalStreamSink:
    def on_start(self, meta: Dict[str, Any]) -> List[bytes]:
        return []

    def on_text_delta(self, text: str) -> List[bytes]:
        return []

    def on_tool_call_start(self, call_id: str, name: str) -> List[bytes]:
        return []

    def on_tool_call_args_delta(self, call_id: str, name: str, delta: str) -> List[bytes]:
        return []

    def on_final(self, meta: Dict[str, Any]) -> List[bytes]:
        return []

    def on_done(self) -> List[bytes]:
        return []

    def flush(self) -> List[bytes]:
        return []


class _OpenAIChatSink(_InternalStreamSink):
    def __init__(self) -> None:
        self.response_id: Optional[str] = None
        self.model: Optional[str] = None
        self.created_at: Optional[int] = None
        self.role_sent = False

    def on_start(self, meta: Dict[str, Any]) -> List[bytes]:
        self.response_id = meta.get("id") or self.response_id or ""
        self.model = meta.get("model") or self.model or ""
        self.created_at = meta.get("created") or meta.get("created_at") or self.created_at or int(time.time())
        if self.role_sent:
            return []
        self.role_sent = True
        return [self._build_chunk({"role": "assistant"})]

    def on_text_delta(self, text: str) -> List[bytes]:
        if not text:
            return []
        return [self._build_chunk({"content": text})]

    def on_tool_call_start(self, call_id: str, name: str) -> List[bytes]:
        if not call_id and not name:
            return []
        return [self._build_chunk({"tool_calls": [self._tool_call_delta(call_id, name, "")]})]

    def on_tool_call_args_delta(self, call_id: str, name: str, delta: str) -> List[bytes]:
        if not delta:
            return []
        return [self._build_chunk({"tool_calls": [self._tool_call_delta(call_id, name, delta)]})]

    def on_final(self, meta: Dict[str, Any]) -> List[bytes]:
        return [self._build_chunk({}, finish_reason=meta.get("finish_reason"), usage=meta.get("usage"))]

    def on_done(self) -> List[bytes]:
        return [_encode_sse("[DONE]")]

    @staticmethod
    def _tool_call_delta(call_id: str, name: str, arguments: str) -> dict:
        # `ChoiceDeltaToolCall.index` is required by OpenAI Python SDK models for streaming chunks.
        # We only support the common case (single tool call at index=0) here.
        return {"index": 0, "id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}

    def _build_chunk(self, delta: dict, finish_reason: Optional[str] = None, usage: Optional[dict] = None) -> bytes:
        chunk = {
            "id": self.response_id or "",
            "object": "chat.completion.chunk",
            "created": self.created_at or int(time.time()),
            "model": self.model or "",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage:
            chunk["usage"] = usage
        return _encode_json(chunk)


class _ResponsesSink(_InternalStreamSink):
    def __init__(self) -> None:
        self.response_id: Optional[str] = None
        self.model: Optional[str] = None
        self.created_at: Optional[int] = None
        self.started = False
        self._text_buf: List[str] = []
        self._tool_calls: Dict[str, Dict[str, str]] = {}  # call_id -> {"name": str, "arguments": str}
        self._seq = 0
        self._msg_item_id = "msg_stream_0"
        self._msg_content_index = 0
        self._tool_item_ids: Dict[str, str] = {}  # call_id -> item_id

    def on_start(self, meta: Dict[str, Any]) -> List[bytes]:
        self.response_id = meta.get("id") or self.response_id or ""
        self.model = meta.get("model") or self.model or ""
        self.created_at = meta.get("created_at") or meta.get("created") or self.created_at or int(time.time())
        if self.started:
            return []
        self.started = True
        resp = self._response_stub(status="in_progress")

        # Compatibility target:
        # - official OpenAI Python SDK `ResponseStreamState` accumulator
        # - formatpack docs (response.output_item.added -> response.content_part.added -> response.output_text.delta)
        outs: List[bytes] = []
        for payload in [
            {"type": "response.created", "sequence_number": self._next_seq(), "response": resp},
            {"type": "response.in_progress", "sequence_number": self._next_seq(), "response": resp},
            {
                "type": "response.output_item.added",
                "sequence_number": self._next_seq(),
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": self._msg_item_id,
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            },
            {
                "type": "response.content_part.added",
                "sequence_number": self._next_seq(),
                "output_index": 0,
                "item_id": self._msg_item_id,
                "content_index": self._msg_content_index,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        ]:
            outs.append(_encode_json(payload, event=payload["type"]))
        return outs

    def on_text_delta(self, text: str) -> List[bytes]:
        if not text:
            return []
        self._text_buf.append(text)
        return [
            _encode_json(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": self._next_seq(),
                    "output_index": 0,
                    "item_id": self._msg_item_id,
                    "content_index": self._msg_content_index,
                    "delta": text,
                }
                ,
                event="response.output_text.delta",
            )
        ]

    def on_tool_call_start(self, call_id: str, name: str) -> List[bytes]:
        if call_id:
            self._tool_calls.setdefault(call_id, {"name": name or "", "arguments": ""})
            if name:
                self._tool_calls[call_id]["name"] = name
            self._tool_item_ids.setdefault(call_id, f"fc_{len(self._tool_item_ids)}")
        item_id = self._tool_item_ids.get(call_id, f"fc_{len(self._tool_item_ids)}")
        self._tool_item_ids[call_id] = item_id
        return [
            _encode_json(
                {
                    "type": "response.output_item.added",
                    "sequence_number": self._next_seq(),
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": name,
                        "arguments": "",
                        "status": "in_progress",
                    },
                }
                ,
                event="response.output_item.added",
            )
        ]

    def on_tool_call_args_delta(self, call_id: str, name: str, delta: str) -> List[bytes]:
        if not delta:
            return []
        if call_id:
            self._tool_calls.setdefault(call_id, {"name": name or "", "arguments": ""})
            if name:
                self._tool_calls[call_id]["name"] = name
            self._tool_calls[call_id]["arguments"] = self._tool_calls[call_id].get("arguments", "") + (delta or "")
        item_id = self._tool_item_ids.get(call_id, "")
        return [
            _encode_json(
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": self._next_seq(),
                    "output_index": 1,
                    "item_id": item_id,
                    "delta": delta,
                }
                ,
                event="response.function_call_arguments.delta",
            )
        ]

    def on_final(self, meta: Dict[str, Any]) -> List[bytes]:
        finish_reason = meta.get("finish_reason") or "stop"
        status = {"stop": "completed", "length": "incomplete", "error": "failed"}.get(finish_reason, "completed")
        resp = self._response_stub(status=status)

        output: List[dict] = []
        # Emit tool calls as output items (best-effort).
        for call_id, entry in self._tool_calls.items():
            output.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": entry.get("name") or "",
                    "arguments": entry.get("arguments") or "",
                    "id": self._tool_item_ids.get(call_id),
                    "status": "completed",
                }
            )

        full_text = "".join(self._text_buf)
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "id": self._msg_item_id,
                "status": "completed",
                "content": [{"type": "output_text", "text": full_text, "annotations": []}],
            }
        )
        resp["output"] = output

        usage = meta.get("usage")
        if usage and isinstance(usage, dict):
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            total_tokens = usage.get("total_tokens")
            if input_tokens is None and "prompt_tokens" in usage:
                input_tokens = usage.get("prompt_tokens", 0)
            if output_tokens is None and "completion_tokens" in usage:
                output_tokens = usage.get("completion_tokens", 0)
            if total_tokens is None and input_tokens is not None and output_tokens is not None:
                total_tokens = int(input_tokens) + int(output_tokens)
            resp["usage"] = {
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
                "total_tokens": int(total_tokens or 0),
            }

        return [_encode_json({"type": "response.completed", "sequence_number": self._next_seq(), "response": resp}, event="response.completed")]

    def on_done(self) -> List[bytes]:
        return [_encode_sse("[DONE]")]

    def _response_stub(self, status: str) -> dict:
        return {
            "object": "response",
            "id": self.response_id or "",
            "model": self.model or "",
            "created_at": self.created_at or int(time.time()),
            "status": status,
            "output": [],
            # Required by OpenAI Responses schema / official SDK.
            "parallel_tool_calls": False,
            "tool_choice": None,
            "tools": [],
        }

    def _next_seq(self) -> int:
        n = int(self._seq)
        self._seq += 1
        return n


class _GeminiSink(_InternalStreamSink):
    def __init__(self) -> None:
        self.response_id: Optional[str] = None
        self.model: Optional[str] = None
        self.tool_args_buffers: Dict[str, str] = {}
        self.tool_names: Dict[str, str] = {}

    def on_start(self, meta: Dict[str, Any]) -> List[bytes]:
        self.response_id = meta.get("id") or self.response_id or "gemini-stream"
        self.model = meta.get("model") or self.model or "gemini"
        return []

    def on_text_delta(self, text: str) -> List[bytes]:
        if not text:
            return []
        payload = {
            "candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "index": 0}],
            "responseId": self.response_id,
            "modelVersion": self.model,
        }
        return [_encode_json(payload)]

    def on_tool_call_start(self, call_id: str, name: str) -> List[bytes]:
        if not call_id:
            return []
        self.tool_args_buffers.setdefault(call_id, "")
        self.tool_names[call_id] = name
        return []

    def on_tool_call_args_delta(self, call_id: str, name: str, delta: str) -> List[bytes]:
        if not call_id:
            return []
        self.tool_args_buffers[call_id] = self.tool_args_buffers.get(call_id, "") + (delta or "")
        self.tool_names[call_id] = name or self.tool_names.get(call_id, "")
        buf = self.tool_args_buffers[call_id].strip()

        args_obj = _safe_json_loads(buf)
        if not isinstance(args_obj, dict) or not args_obj:
            return []

        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"functionCall": {"id": call_id, "name": self.tool_names.get(call_id, name) or "", "args": args_obj}}],
                        "role": "model",
                    },
                    "index": 0,
                }
            ],
            "responseId": self.response_id,
            "modelVersion": self.model,
        }
        self.tool_args_buffers[call_id] = ""
        return [_encode_json(payload)]

    def flush(self) -> List[bytes]:
        outs: List[bytes] = []
        for call_id, buf in list(self.tool_args_buffers.items()):
            b = (buf or "").strip()
            if not b:
                continue
            args_obj = _safe_json_loads(b)
            if isinstance(args_obj, dict) and args_obj:
                outs.extend(self.on_tool_call_args_delta(call_id, self.tool_names.get(call_id, ""), b))
        return outs

    def on_done(self) -> List[bytes]:
        return [_encode_sse("[DONE]")]


class _ClaudeSink(_InternalStreamSink):
    def __init__(self) -> None:
        self.message_id: Optional[str] = None
        self.model: Optional[str] = None
        self.started = False
        self.open_blocks: List[Tuple[int, str]] = []  # (index, type)
        self.next_block_index = 0
        self.tool_names: Dict[str, str] = {}

    def on_start(self, meta: Dict[str, Any]) -> List[bytes]:
        if self.started:
            return []
        self.started = True
        self.message_id = meta.get("id") or self.message_id or "claude-stream"
        self.model = meta.get("model") or self.model or "claude"
        payload = {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        return [_encode_json(payload, event="message_start")]

    def _ensure_text_block(self) -> List[bytes]:
        for _, btype in self.open_blocks:
            if btype == "text":
                return []
        index = self.next_block_index
        self.next_block_index += 1
        self.open_blocks.append((index, "text"))
        payload = {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}}
        return [_encode_json(payload, event="content_block_start")]

    def on_text_delta(self, text: str) -> List[bytes]:
        if not text:
            return []
        outs: List[bytes] = []
        outs.extend(self._ensure_text_block())
        index = next((i for i, t in self.open_blocks if t == "text"), 0)
        payload = {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}}
        outs.append(_encode_json(payload, event="content_block_delta"))
        return outs

    def on_tool_call_start(self, call_id: str, name: str) -> List[bytes]:
        if not call_id:
            return []
        index = self.next_block_index
        self.next_block_index += 1
        self.open_blocks.append((index, f"tool_use:{call_id}"))
        self.tool_names[call_id] = name
        payload = {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}},
        }
        return [_encode_json(payload, event="content_block_start")]

    def on_tool_call_args_delta(self, call_id: str, name: str, delta: str) -> List[bytes]:
        if not call_id or not delta:
            return []
        self.tool_names[call_id] = name or self.tool_names.get(call_id, "")
        index = next((i for i, t in self.open_blocks if t == f"tool_use:{call_id}"), None)
        outs: List[bytes] = []
        if index is None:
            outs.extend(self.on_tool_call_start(call_id, self.tool_names.get(call_id, "")))
            index = next((i for i, t in self.open_blocks if t == f"tool_use:{call_id}"), 0)
        payload = {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": delta}}
        outs.append(_encode_json(payload, event="content_block_delta"))
        return outs

    def on_final(self, meta: Dict[str, Any]) -> List[bytes]:
        outs: List[bytes] = []
        for idx, _ in list(self.open_blocks):
            outs.append(_encode_json({"type": "content_block_stop", "index": idx}, event="content_block_stop"))
        self.open_blocks = []

        finish_reason = meta.get("finish_reason")
        # Map OpenAI-style finish reasons to Anthropic stop_reason values.
        # Anthropic: end_turn | max_tokens | tool_use | stop_sequence | ...
        stop_reason = "end_turn"
        if finish_reason in (None, "", "stop"):
            stop_reason = "end_turn"
        elif finish_reason in ("tool_calls", "function_call"):
            stop_reason = "tool_use"
        elif finish_reason in ("length",):
            stop_reason = "max_tokens"
        else:
            # Best-effort passthrough if already Anthropic-like.
            stop_reason = str(finish_reason)

        usage = meta.get("usage")
        usage_obj: Optional[dict] = None
        if isinstance(usage, dict):
            # Anthropic streams report cumulative output_tokens in message_delta.
            # We may not have exact token counts here; use any known fields, else 0.
            out_toks = usage.get("output_tokens")
            if out_toks is None and "completion_tokens" in usage:
                out_toks = usage.get("completion_tokens")
            if out_toks is None and "total_tokens" in usage and "input_tokens" in usage:
                try:
                    out_toks = int(usage["total_tokens"]) - int(usage["input_tokens"])
                except Exception:
                    out_toks = None
            try:
                usage_obj = {"output_tokens": int(out_toks or 0)}
            except Exception:
                usage_obj = {"output_tokens": 0}
        else:
            usage_obj = {"output_tokens": 0}

        delta_payload: dict = {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": usage_obj}
        outs.append(_encode_json(delta_payload, event="message_delta"))
        outs.append(_encode_json({"type": "message_stop"}, event="message_stop"))
        return outs

    def on_done(self) -> List[bytes]:
        return [_encode_sse("[DONE]")]


def _create_sink(to_format: str) -> _InternalStreamSink:
    if to_format == "openai_chat":
        return _OpenAIChatSink()
    if to_format == "openai_responses":
        return _ResponsesSink()
    if to_format == "gemini_chat":
        return _GeminiSink()
    if to_format == "claude_chat":
        return _ClaudeSink()
    return _OpenAIChatSink()


def _decode_to_internal(
    from_format: str,
    event_name: Optional[str],
    event: Dict[str, Any],
    meta: Dict[str, Any],
    seen_tool_calls: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    outs: List[Dict[str, Any]] = []

    def emit_start_once() -> None:
        if meta and "__started__" not in seen_tool_calls:
            seen_tool_calls["__started__"] = {"started": True}
            outs.append({"type": "start", "meta": dict(meta)})

    if from_format == "openai_chat":
        if event.get("choices") is None:
            return []
        meta["id"] = event.get("id", meta.get("id"))
        meta["model"] = event.get("model", meta.get("model"))
        meta["created"] = event.get("created", meta.get("created")) or int(time.time())
        emit_start_once()

        choice = (event.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content:
            outs.append({"type": "text_delta", "text": content})

        for tc in delta.get("tool_calls") or []:
            call_id = tc.get("id") or ""
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            args = fn.get("arguments") or ""
            if call_id and call_id not in seen_tool_calls:
                seen_tool_calls[call_id] = {"name": name}
                outs.append({"type": "tool_call_start", "id": call_id, "name": name})
            if args:
                outs.append({"type": "tool_call_args_delta", "id": call_id, "name": name, "delta": args})

        finish_reason = choice.get("finish_reason")
        if finish_reason:
            outs.append({"type": "final", "meta": {"finish_reason": finish_reason, "usage": event.get("usage")}})
            outs.append({"type": "done"})

        return outs

    if from_format == "openai_responses":
        etype = event.get("type")
        if not etype:
            return []
        resp = event.get("response") or {}
        item_map = seen_tool_calls.setdefault("__responses_item_id_map__", {})
        if etype in {"response.created", "response.in_progress"}:
            meta["id"] = resp.get("id", meta.get("id"))
            meta["model"] = resp.get("model", meta.get("model"))
            meta["created_at"] = resp.get("created_at", meta.get("created_at")) or int(time.time())
            emit_start_once()
            return outs

        if etype == "response.output_text.delta":
            text = event.get("delta") or event.get("text") or ""
            if text:
                outs.append({"type": "text_delta", "text": text})
            return outs

        if etype == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                item_id = item.get("id") or ""
                call_id = item.get("call_id") or item_id or ""
                name = item.get("name") or ""
                if call_id and call_id not in seen_tool_calls:
                    seen_tool_calls[call_id] = {"name": name}
                    outs.append({"type": "tool_call_start", "id": call_id, "name": name})
                if item_id and isinstance(item_map, dict):
                    item_map[item_id] = call_id
            return outs

        if etype in {"response.function_call_arguments.delta", "response.function_call_arguments.done", "response.function_call.delta"}:
            item_id = event.get("item_id") or ""
            call_id = ""
            if item_id and isinstance(item_map, dict):
                call_id = item_map.get(item_id, "") or ""
            name = seen_tool_calls.get(call_id, {}).get("name") or ""
            delta_args = event.get("delta") or event.get("arguments") or ""
            if call_id and call_id not in seen_tool_calls:
                seen_tool_calls[call_id] = {"name": name}
                outs.append({"type": "tool_call_start", "id": call_id, "name": name})
            if delta_args and call_id:
                outs.append({"type": "tool_call_args_delta", "id": call_id, "name": name, "delta": delta_args})
            return outs

        if etype in {"response.completed", "response.failed", "response.incomplete", "error"}:
            usage = None
            finish_reason = "stop"
            if etype == "error":
                finish_reason = "error"
            else:
                status = (resp.get("status") or "").lower()
                if status == "completed":
                    finish_reason = "stop"
                elif status == "incomplete":
                    finish_reason = "length"
                elif status == "failed":
                    finish_reason = "error"

            if isinstance(resp.get("usage"), dict):
                u = resp["usage"]
                usage = {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0), "total_tokens": u.get("total_tokens", 0)}

            outs.append({"type": "final", "meta": {"finish_reason": finish_reason, "usage": usage}})
            outs.append({"type": "done"})
            return outs

        return outs

    if from_format == "gemini_chat":
        candidates = event.get("candidates") or []
        if not isinstance(candidates, list) or not candidates:
            return []
        meta.setdefault("id", event.get("responseId") or event.get("id") or meta.get("id") or "gemini-stream")
        meta.setdefault("model", event.get("modelVersion") or meta.get("model") or "gemini")
        emit_start_once()

        candidate = candidates[0] if candidates else {}
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if isinstance(part, dict) and "text" in part and part["text"]:
                outs.append({"type": "text_delta", "text": part.get("text")})
            if isinstance(part, dict) and "functionCall" in part:
                fc = part.get("functionCall") or {}
                call_id = fc.get("id") or f"gemini_call_{len([k for k in seen_tool_calls.keys() if k != '__started__'])}"
                name = fc.get("name") or ""
                args_obj = fc.get("args") or {}
                if call_id not in seen_tool_calls:
                    seen_tool_calls[call_id] = {"name": name}
                    outs.append({"type": "tool_call_start", "id": call_id, "name": name})
                outs.append({"type": "tool_call_args_delta", "id": call_id, "name": name, "delta": _json_dumps_compact(args_obj)})
        return outs

    if from_format == "claude_chat":
        dtype = event.get("type") or event_name
        if not dtype:
            return []

        if dtype == "message_start":
            msg = event.get("message") or {}
            meta["id"] = msg.get("id", meta.get("id"))
            meta["model"] = msg.get("model", meta.get("model"))
            emit_start_once()

            for b in msg.get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    call_id = b.get("id") or ""
                    name = b.get("name") or ""
                    if call_id and call_id not in seen_tool_calls:
                        seen_tool_calls[call_id] = {"name": name}
                        outs.append({"type": "tool_call_start", "id": call_id, "name": name})
                    input_obj = b.get("input") or {}
                    if input_obj:
                        outs.append({"type": "tool_call_args_delta", "id": call_id, "name": name, "delta": _json_dumps_compact(input_obj)})
            return outs

        if dtype == "content_block_start":
            cb = event.get("content_block") or {}
            if cb.get("type") == "tool_use":
                call_id = cb.get("id") or ""
                name = cb.get("name") or ""
                if call_id and call_id not in seen_tool_calls:
                    seen_tool_calls[call_id] = {"name": name}
                    outs.append({"type": "tool_call_start", "id": call_id, "name": name})
                input_obj = cb.get("input") or {}
                if input_obj:
                    outs.append({"type": "tool_call_args_delta", "id": call_id, "name": name, "delta": _json_dumps_compact(input_obj)})
            return outs

        if dtype == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                text = delta.get("text") or ""
                if text:
                    outs.append({"type": "text_delta", "text": text})
            elif delta.get("type") == "input_json_delta":
                partial = delta.get("partial_json") or ""
                last_tool = next((k for k in reversed(list(seen_tool_calls.keys())) if k != "__started__"), None)
                if last_tool:
                    outs.append({"type": "tool_call_args_delta", "id": last_tool, "name": seen_tool_calls.get(last_tool, {}).get("name") or "", "delta": partial})
            return outs

        if dtype == "message_delta":
            delta = event.get("delta") or {}
            outs.append({"type": "final", "meta": {"finish_reason": delta.get("stop_reason"), "usage": event.get("usage")}})
            return outs

        if dtype == "message_stop":
            outs.append({"type": "done"})
            return outs

        return outs

    return outs


class _StreamTranscoder(BaseSSEAdapter):
    def __init__(self, from_format: str, to_format: str) -> None:
        self.from_format = from_format
        self.to_format = to_format
        self.started = False
        self.meta: Dict[str, Any] = {}
        self.seen_tool_calls: Dict[str, Dict[str, Any]] = {}
        self.sink = _create_sink(to_format)

    def handle_event(self, event_name: Optional[str], event: dict) -> List[bytes]:
        outs: List[bytes] = []
        internal_events = _decode_to_internal(self.from_format, event_name, event, self.meta, self.seen_tool_calls)

        for ie in internal_events:
            itype = ie.get("type")
            if itype == "start":
                self.started = True
                outs.extend(self.sink.on_start(ie.get("meta") or {}))
            elif itype == "text_delta":
                if not self.started:
                    self.started = True
                    outs.extend(self.sink.on_start(self.meta))
                outs.extend(self.sink.on_text_delta(ie.get("text") or ""))
            elif itype == "tool_call_start":
                if not self.started:
                    self.started = True
                    outs.extend(self.sink.on_start(self.meta))
                outs.extend(self.sink.on_tool_call_start(ie.get("id") or "", ie.get("name") or ""))
            elif itype == "tool_call_args_delta":
                if not self.started:
                    self.started = True
                    outs.extend(self.sink.on_start(self.meta))
                outs.extend(self.sink.on_tool_call_args_delta(ie.get("id") or "", ie.get("name") or "", ie.get("delta") or ""))
            elif itype == "final":
                if not self.started:
                    self.started = True
                    outs.extend(self.sink.on_start(self.meta))
                outs.extend(self.sink.on_final(ie.get("meta") or {}))
            elif itype == "done":
                outs.extend(self.sink.on_done())

        return outs

    def on_done(self) -> List[bytes]:
        return self.sink.on_done()

    def flush(self) -> List[bytes]:
        return self.sink.flush()


def create_stream_transformer(from_format: str, to_format: str) -> Optional[SSEBufferTransformer]:
    """创建指定格式之间的流式转换器"""
    if not from_format or not to_format or from_format == to_format:
        return None
    supported = {"openai_chat", "openai_responses", "gemini_chat", "claude_chat"}
    if from_format not in supported or to_format not in supported:
        return None
    return SSEBufferTransformer(_StreamTranscoder(from_format, to_format))
