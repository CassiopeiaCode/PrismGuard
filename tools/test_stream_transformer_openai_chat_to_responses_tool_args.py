import json
import unittest

from ai_proxy.proxy.stream_transformer import create_stream_transformer


def _iter_sse_frames(b: bytes):
    text = b.decode("utf-8", errors="ignore")
    for raw in text.split("\n\n"):
        raw = raw.strip()
        if not raw:
            continue
        event = None
        data_lines = []
        for line in raw.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
        data = "\n".join(data_lines)
        yield event, data


class TestStreamTransformerOpenAIChatToResponsesToolArgs(unittest.TestCase):
    def test_openai_chat_tool_args_missing_id_name_chunks(self) -> None:
        tr = create_stream_transformer("openai_chat", "openai_responses")
        self.assertIsNotNone(tr)

        def sse_data(obj):
            return ("data: " + json.dumps(obj, separators=(",", ":")) + "\n\n").encode("utf-8")

        # Tool call where only the first chunk has id+name; later chunks omit both.
        chunks = [
            sse_data(
                {
                    "id": "chatcmpl_x",
                    "object": "chat.completion.chunk",
                    "created": 1710000000,
                    "model": "gpt-5.2",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "Bash", "arguments": ""},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            ),
            sse_data(
                {
                    "id": "chatcmpl_x",
                    "object": "chat.completion.chunk",
                    "created": 1710000000,
                    "model": "gpt-5.2",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "type": "function", "function": {"arguments": "{\"cmd\":\"ls\""}}
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            ),
            sse_data(
                {
                    "id": "chatcmpl_x",
                    "object": "chat.completion.chunk",
                    "created": 1710000000,
                    "model": "gpt-5.2",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [{"index": 0, "type": "function", "function": {"arguments": "}"}}]},
                            "finish_reason": None,
                        }
                    ],
                }
            ),
            sse_data(
                {
                    "id": "chatcmpl_x",
                    "object": "chat.completion.chunk",
                    "created": 1710000000,
                    "model": "gpt-5.2",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                }
            ),
            b"data: [DONE]\n\n",
        ]

        out_frames = []
        for c in chunks:
            for out in tr.feed(c):
                for ev, data in _iter_sse_frames(out):
                    if not data or data == "[DONE]":
                        continue
                    out_frames.append((ev, json.loads(data)))
        for out in tr.flush():
            for ev, data in _iter_sse_frames(out):
                if not data or data == "[DONE]":
                    continue
                out_frames.append((ev, json.loads(data)))

        # Ensure we emitted a function_call item and at least one arguments delta with an item_id.
        fc_added = next(
            (
                obj
                for ev, obj in out_frames
                if obj.get("type") == "response.output_item.added"
                and obj.get("item", {}).get("type") == "function_call"
                and obj.get("item", {}).get("call_id") == "call_1"
            ),
            None,
        )
        self.assertIsNotNone(fc_added)

        arg_deltas = [
            obj
            for ev, obj in out_frames
            if obj.get("type") == "response.function_call_arguments.delta" and (obj.get("item_id") or "")
        ]
        self.assertTrue(arg_deltas, "expected response.function_call_arguments.delta with item_id")
        joined = "".join(obj.get("delta") or "" for obj in arg_deltas)
        self.assertEqual(joined, "{\"cmd\":\"ls\"}")

        completed = next((obj for ev, obj in out_frames if obj.get("type") == "response.completed"), None)
        self.assertIsNotNone(completed)
        output = (completed.get("response") or {}).get("output") or []
        fc = next((it for it in output if it.get("type") == "function_call" and it.get("call_id") == "call_1"), None)
        self.assertIsNotNone(fc)
        self.assertEqual(fc.get("arguments"), "{\"cmd\":\"ls\"}")

