"""
流式响应检查工具
用于在发送响应头之前检查流式内容是否有效（避免返回空回复）
"""
import json
from typing import Tuple, List, Optional, Dict, Any

class StreamChecker:
    """流式内容检查器"""
    
    def __init__(self, format_name: str):
        self.format_name = format_name
        self.accumulated_content = ""
        self.has_tool_call = False
        self.char_threshold = 2
        self.accumulated_bytes = b""  # 用于累积 Gemini 的不完整 JSON
        
    def check_chunk(self, chunk: bytes) -> bool:
        """
        检查 chunk 数据
        Returns:
            bool: 是否已满足放行条件（内容>2chars 或 有工具调用）
        """
        # 先打印调试信息再检查条件
        print(f"[DEBUG] check_chunk called: format={self.format_name}, chunk_size={len(chunk)}, chunk_preview={chunk[:100].__repr__()}")
        
        if self.has_tool_call or len(self.accumulated_content) > self.char_threshold:
            print(f"[DEBUG] Already satisfied: has_tool_call={self.has_tool_call}, content_len={len(self.accumulated_content)}")
            return True
        
        try:
            text = chunk.decode("utf-8")
            print(f"[DEBUG] Decoded chunk text (first 200 chars): {text[:200]}")
        except UnicodeDecodeError as e:
            # 忽略解码错误的块（可能是被截断的多字节字符）
            print(f"[DEBUG] UnicodeDecodeError: {e}")
            return False
        
        # 检测格式：Gemini 使用 SSE，OpenAI/Claude 也使用 SSE
        if self.format_name == "gemini_chat":
            result = self._check_gemini_format(text)
            print(f"[DEBUG] Gemini format check result: {result}")
            return result
        else:
            result = self._check_sse_format(text)
            print(f"[DEBUG] SSE format check result: {result}")
            return result
    
    def _check_sse_format(self, text: str) -> bool:
        """检查 SSE 格式（OpenAI/Claude）"""
        for line in text.split('\n'):
            line = line.strip()
            if not line.startswith('data: '):
                continue
                
            data_str = line[6:]  # remove 'data: '
            if data_str == '[DONE]':
                continue
                
            try:
                data = json.loads(data_str)
                self._parse_data(data)
                
                if self.has_tool_call or len(self.accumulated_content) > self.char_threshold:
                    return True
            except json.JSONDecodeError:
                continue
                
        return False
    
    def _check_gemini_format(self, text: str) -> bool:
        """
        检查 Gemini SSE 流式格式
        
        Gemini SSE 响应格式（使用 alt=sse 参数）：
        data: {"candidates": [...], ...}
        
        data: {"candidates": [...], ...}
        
        每行以 "data: " 开头，后面是完整的 JSON 对象
        """
        print(f"[DEBUG] _check_gemini_format (SSE): processing {len(text)} chars")
        print(f"[DEBUG] Gemini SSE text preview: {text[:300]}")
        
        # 按行处理 SSE 格式
        for line in text.split('\n'):
            line = line.strip()
            
            # 跳过空行
            if not line:
                continue
            
            # 检查是否是 SSE 数据行
            if not line.startswith('data: '):
                print(f"[DEBUG] Skipping non-data line: {line[:50]}")
                continue
            
            # 提取 JSON 数据（移除 "data: " 前缀）
            data_str = line[6:]  # remove 'data: '
            
            # 跳过 [DONE] 标记
            if data_str == '[DONE]':
                print(f"[DEBUG] Found [DONE] marker")
                continue
            
            try:
                data = json.loads(data_str)
                print(f"[DEBUG] Gemini SSE parsed JSON object")
                
                self._parse_gemini_data(data)
                
                if self.has_tool_call or len(self.accumulated_content) > self.char_threshold:
                    print(f"[DEBUG] Gemini SSE check satisfied: has_tool_call={self.has_tool_call}, content_len={len(self.accumulated_content)}")
                    return True
            except json.JSONDecodeError as e:
                print(f"[DEBUG] JSON parse error: {e}, data_str: {data_str[:100]}")
                continue
        
        print(f"[DEBUG] Gemini SSE check not satisfied yet: content_len={len(self.accumulated_content)}")
        return False

    def _parse_data(self, data: dict):
        """解析单条数据"""
        # OpenAI Chat 和 Codex 格式
        if "choices" in data and isinstance(data["choices"], list):
            for choice in data["choices"]:
                # OpenAI Chat: 使用 delta.content
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    self.accumulated_content += content
                
                # OpenAI Codex/Completions: 使用 text 字段
                text = choice.get("text")
                if text:
                    self.accumulated_content += text
                
                # 检查工具调用（仅 Chat 格式支持）
                if delta.get("tool_calls"):
                    self.has_tool_call = True
                    
        # Claude 格式 (假设通过 upstream 已经是转为兼容格式，或者透传)
        # 如果是透传的 Claude 原生 SSE，它是 event: ... data: ...
        # 这里简化处理，尝试识别常见的 type
        if "type" in data:
            dtype = data["type"]
            if dtype == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    self.accumulated_content += delta.get("text", "")
            elif dtype == "message_start":
                # 检查 message 中的 content 是否有 tool_use
                msg = data.get("message", {})
                for c in msg.get("content", []):
                    if c.get("type") == "tool_use":
                        self.has_tool_call = True
            elif dtype == "content_block_start":
                 if data.get("content_block", {}).get("type") == "tool_use":
                     self.has_tool_call = True
    
    def _parse_gemini_data(self, data: dict):
        """解析 Gemini 格式的数据"""
        print(f"[DEBUG] _parse_gemini_data: {data}")  # 修复 f-string
        candidates = data.get("candidates", [])
        if not candidates:
            print(f"[DEBUG] No candidates in Gemini data")
            return
        
        print(f"[DEBUG] Processing {len(candidates)} candidates")
        
        for candidate in candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            print(f"[DEBUG] Candidate has {len(parts)} parts")
            
            for part in parts:
                # 检查文本内容
                if "text" in part:
                    text_content = part.get("text", "")
                    print(f"[DEBUG] Found text in part: {text_content[:100]}")
                    self.accumulated_content += text_content
                
                # 检查函数调用（Gemini 的工具调用）
                if "functionCall" in part:
                    print(f"[DEBUG] Found functionCall in part")
                    self.has_tool_call = True


def check_response_content(response: Dict[str, Any], format_name: str) -> Tuple[bool, Optional[str]]:
    """
    检查非流式响应内容是否满足条件
    
    Args:
        response: 响应体（JSON 格式）
        format_name: 响应格式名称
    
    Returns:
        (是否通过, 错误消息)
    """
    accumulated_content = ""
    has_tool_call = False
    char_threshold = 2
    
    try:
        # 检查响应是否为字典类型
        if not isinstance(response, dict):
            print(f"[WARN] Response is not a dict: {type(response)}, allowing it to pass")
            return True, None
        
        if format_name == "gemini_chat":
            # Gemini 格式
            candidates = response.get("candidates", [])
            for candidate in candidates:
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                
                for part in parts:
                    if "text" in part:
                        accumulated_content += part.get("text", "")
                    if "functionCall" in part:
                        has_tool_call = True
        
        elif format_name == "openai_codex":
            # OpenAI Codex/Completions 格式
            choices = response.get("choices", [])
            for choice in choices:
                text = choice.get("text", "")
                if text:
                    accumulated_content += text
        
        elif format_name == "claude_chat":
            # Claude 格式
            content_blocks = response.get("content", [])
            for block in content_blocks:
                if block.get("type") == "text":
                    accumulated_content += block.get("text", "")
                elif block.get("type") == "tool_use":
                    has_tool_call = True
        
        else:
            # OpenAI Chat 格式（默认）
            choices = response.get("choices", [])
            for choice in choices:
                message = choice.get("message", {})
                content = message.get("content", "")
                if content:
                    accumulated_content += content
                
                # 检查工具调用
                if message.get("tool_calls"):
                    has_tool_call = True
        
        # 检查是否满足条件
        if has_tool_call or len(accumulated_content) > char_threshold:
            return True, None
        
        # 不满足条件
        error_msg = (
            f"Response content validation failed: "
            f"accumulated {len(accumulated_content)} chars (threshold: {char_threshold}), "
            f"has_tool_call: {has_tool_call}. "
            f"The AI response appears to be empty or too short."
            f"Format name: {format_name}"
        )
        return False, error_msg
    
    except Exception as e:
        # 解析错误时，保守处理：允许通过
        print(f"[WARN] Response content check error: {e}")
        return True, None
