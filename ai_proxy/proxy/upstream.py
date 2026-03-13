"""
上游 HTTP 客户端封装 - 修复版
"""
import httpx
import json
import gzip
from typing import Optional, Dict, Any, AsyncIterator
from fastapi.responses import StreamingResponse, JSONResponse, Response
from ai_proxy.utils.memory_guard import check_container
from ai_proxy.proxy.stream_checker import StreamChecker, check_response_content
from ai_proxy.proxy.stream_transformer import create_stream_transformer
import asyncio

# 全局 HTTP 客户端池（每个 base_url 一个客户端）
_client_pool: Dict[str, httpx.AsyncClient] = {}


def get_or_create_client(base_url: str) -> httpx.AsyncClient:
    """获取或创建 HTTP 客户端（复用连接池）"""
    if base_url not in _client_pool:
        _client_pool[base_url] = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(
                max_keepalive_connections=200,
                max_connections=1000,
                keepalive_expiry=30.0
            )
        )
    
    # 定期检查客户端池
    check_container(_client_pool, "http_client_pool")
    
    return _client_pool[base_url]


async def cleanup_clients():
    """清理所有客户端（应用关闭时调用）"""
    for client in _client_pool.values():
        await client.aclose()
    _client_pool.clear()


class UpstreamClient:
    """上游服务客户端"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = get_or_create_client(self.base_url)  # ✅ 复用客户端
    
    async def forward_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[Dict[str, Any]] = None,
        is_stream: bool = False,
        src_format: Optional[str] = None,
        target_format: Optional[str] = None,
        delay_stream_header: bool = False
    ):
        """
        转发请求到上游
        
        Args:
            src_format: 客户端原始格式（用于响应转换）
            target_format: 上游API格式（响应需要从此格式转换回 src_format）
            delay_stream_header: 是否延迟发送流式响应头（直到内容>2chars或有工具调用）
        """
        # 过滤掉不需要的头，并移除 Accept-Encoding
        filtered_headers = {
            k: v for k, v in headers.items()
            if k.lower() not in ["host", "content-length", "accept-encoding"]
        }
        # 明确禁止压缩，避免编码问题
        filtered_headers["Accept-Encoding"] = "identity"
        print(f"[UPSTREAM] Request headers: {filtered_headers}")
        
        url = f"{self.base_url}{path}"
        
        # 🔥 强制为 Gemini 流式请求添加 alt=sse 参数
        if target_format == "gemini_chat" and is_stream and "streamGenerateContent" in path:
            if "alt=sse" not in url:
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}alt=sse"
                print(f"[UPSTREAM] OK Added alt=sse parameter for Gemini streaming: {url}")
        
        try:
            if is_stream:
                # 流式请求 - 使用手动请求管理以支持延迟响应头
                req = self.client.build_request(
                    method,
                    url,
                    headers=filtered_headers,
                    json=body
                )
                
                response = await self.client.send(req, stream=True)
                
                # 打印上游响应头信息
                print(f"[UPSTREAM] Upstream response status: {response.status_code}")
                print(f"[UPSTREAM] Upstream response headers: {dict(response.headers)}")
                print(f"[UPSTREAM] Upstream Content-Type: {response.headers.get('content-type', 'None')}")
                print(f"[UPSTREAM] Upstream Content-Encoding: {response.headers.get('content-encoding', 'None')}")
                
                if response.status_code != 200:
                    # 非 200：不要强制 JSON，也不要提前 aclose()
                    # 需求：透传上游状态码 + 上游响应体（尽可能保留 content-type）
                    filtered_header_names = [
                        "content-length", "transfer-encoding", "content-encoding",
                        "set-cookie",
                        "strict-transport-security",
                        "content-security-policy", "content-security-policy-report-only",
                        "x-frame-options",
                        "x-content-type-options",
                        "x-xss-protection",
                        "permissions-policy",
                        "referrer-policy",
                    ]
                    pass_headers = {
                        k: v for k, v in response.headers.items()
                        if k.lower() not in filtered_header_names
                    }

                    try:
                        err_body = await response.aread()
                    finally:
                        await response.aclose()

                    media_type = response.headers.get("content-type")
                    return Response(
                        content=err_body,
                        status_code=response.status_code,
                        headers=pass_headers,
                        media_type=media_type,
                    )

                # 启用延迟检查
                if delay_stream_header:
                    print(f"[UPSTREAM] delay_stream_header enabled, target_format={target_format}")
                    checker = StreamChecker(target_format or "openai_chat")
                    buffer = []
                    valid = False
                    chunk_count = 0
                    
                    try:
                        # 创建迭代器
                        aiter = response.aiter_bytes()
                        
                        print(f"[UPSTREAM] Starting stream pre-read loop... {target_format}")
                        # 预读循环 - 使用迭代器
                        while True:
                            try:
                                chunk = await aiter.__anext__()
                                chunk_count += 1
                                print(f"[UPSTREAM] Received chunk #{chunk_count}, size={len(chunk)}")
                                
                                # 打印 chunk 的内容（前500字符）
                                try:
                                    chunk_text = chunk.decode('utf-8')
                                    print(f"[UPSTREAM] Chunk content: {chunk_text[:500]}")
                                except:
                                    print(f"[UPSTREAM] Chunk raw bytes: {chunk[:100]}")
                                
                                buffer.append(chunk)
                                
                                check_result = checker.check_chunk(chunk)
                                print(f"[UPSTREAM] check_chunk returned: {check_result}")
                                
                                if check_result:
                                    valid = True
                                    print(f"[UPSTREAM] Stream validation passed after {chunk_count} chunks, total_bytes={sum(len(b) for b in buffer)}")
                                    break
                                
                                # 保护性限制：如果超过 1KB 还没满足条件，强制放行
                                current_size = sum(len(b) for b in buffer)
                                if current_size > 1048:
                                    print(f"[UPSTREAM] Protection limit: forcing pass after {current_size} bytes in {chunk_count} chunks")
                                    valid = True # 视为通过，避免一直卡住
                                    break
                            except StopAsyncIteration:
                                print(f"[UPSTREAM] Stream ended during pre-read")
                                break
                        
                        print(f"[UPSTREAM] Stream pre-read loop ended: valid={valid}, chunk_count={chunk_count}, total_bytes={sum(len(b) for b in buffer)}")
                        
                        # 如果循环结束（流结束了）还没有 valid，检查一下
                        # 此时 valid 仍为 False，buffer 包含所有数据
                        if not valid:
                            # 检查 buffer 是否为空或内容不足
                            total_bytes = sum(len(b) for b in buffer)
                            if total_bytes == 0:
                                # 完全没有接收到任何数据
                                print(f"[UPSTREAM] ERROR: Stream ended without any content")
                                raise Exception("Stream ended without any content")
                            else:
                                # 接收到了数据但不满足验证条件（内容太少或格式不对）
                                print(f"[UPSTREAM] ERROR: Stream validation failed after receiving {total_bytes} bytes in {chunk_count} chunks")
                                raise Exception(f"Stream content validation failed: received {total_bytes} bytes but content is insufficient, debug:{buffer.__repr__()}")
                            
                    except Exception as e:
                        print(f"[UPSTREAM] STREAM_PRE_READ_ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        await response.aclose()

                        # delay_stream_header 场景下：响应头尚未发送。
                        # 这里发生错误说明“开始流之前就失败了”，因此统一返回普通 JSON 错误体（不返回 SSE）。
                        # 注意：OpenAI 的 HTTP 错误响应通常是 {"error": {...}}（与是否为 Responses/Chat 无关）。
                        error_format = src_format or target_format or "openai_chat"

                        err_obj = {
                            "code": "UPSTREAM_STREAM_ERROR",
                            "message": f"Stream disconnected before valid content: {str(e)}",
                            "type": "upstream_stream_error",
                            "format": error_format,
                        }

                        # 预读阶段断流/校验失败：按要求使用 429
                        return JSONResponse(
                            status_code=429,
                            content={"error": err_obj},
                        )

                    # 构造新的生成器，先发 buffer，再发剩余流
                    print(f"[UPSTREAM] Creating combined generator with {len(buffer)} buffered chunks")
                    print(f"[UPSTREAM] Buffer content preview: {buffer[0][:200] if buffer else 'empty'}")
                    
                    # 🔥 SSE 格式不需要 gzip 压缩（HTTP 层面会自动处理）
                    use_gzip = False
                    
                    # 检查是否需要流式响应转换
                    if src_format and target_format and src_format != target_format:
                        print(f"[UPSTREAM] Stream response transform enabled: {target_format} -> {src_format}")
                        combined_gen = self._create_combined_generator_with_transform(
                            buffer, aiter, response, target_format, src_format, use_gzip
                        )
                    else:
                        print(f"[UPSTREAM] No stream response transform needed")
                        combined_gen = self._create_combined_generator(buffer, aiter, response, use_gzip)

                    print(f"[UPSTREAM] All response headers: {dict(response.headers)}")
                    
                    # 构建要传递的响应头
                    filtered_header_names = [
                        "content-length", "transfer-encoding", "content-encoding",
                        "set-cookie",
                        "strict-transport-security",
                        "content-security-policy", "content-security-policy-report-only",
                        "x-frame-options",
                        "x-content-type-options",
                        "x-xss-protection",
                        "permissions-policy",
                        "referrer-policy",
                    ]
                    pass_headers = {k: v for k, v in response.headers.items()
                                   if k.lower() not in filtered_header_names}
                    
                    # 🔥 确保 Gemini SSE 格式的 Content-Type 正确
                    # 使用 alt=sse 参数后，上游应该返回 text/event-stream
                    if target_format == "gemini_chat":
                        if pass_headers.get("content-type") != "text/event-stream":
                            print(f"[UPSTREAM] WARN Setting Content-Type to text/event-stream for Gemini SSE format")
                            pass_headers["content-type"] = "text/event-stream"
                        else:
                            print(f"[UPSTREAM] OK Content-Type is already text/event-stream")
                    
                    print(f"[UPSTREAM] Passing headers (after filtering): {pass_headers}")
                    
                    streaming_resp = StreamingResponse(
                        combined_gen,
                        headers=pass_headers
                    )
                    print(f"[UPSTREAM] StreamingResponse object created, returning to caller")
                    return streaming_resp
                
                else:
                    # 不延迟，直接透传
                    filtered_header_names = [
                        "content-length", "transfer-encoding", "content-encoding",
                        "set-cookie",
                        "strict-transport-security",
                        "content-security-policy", "content-security-policy-report-only",
                        "x-frame-options",
                        "x-content-type-options",
                        "x-xss-protection",
                        "permissions-policy",
                        "referrer-policy",
                    ]
                    
                    # 准备响应头
                    pass_headers = {k: v for k, v in response.headers.items()
                                   if k.lower() not in filtered_header_names}
                    
                    # 🔥 确保 Gemini SSE 格式的 Content-Type 正确
                    if target_format == "gemini_chat":
                        if pass_headers.get("content-type") != "text/event-stream":
                            print(f"[UPSTREAM] WARN Setting Content-Type to text/event-stream for Gemini SSE format")
                            pass_headers["content-type"] = "text/event-stream"
                        else:
                            print(f"[UPSTREAM] OK Content-Type is already text/event-stream")

                    # ✅ 即使不启用 delay_stream_header，也要支持流式响应格式转换
                    # 这里必须使用自定义生成器来确保 aclose() 被调用，并且可做 SSE 转换。
                    use_gzip = False  # SSE 不做应用层 gzip
                    aiter = response.aiter_bytes()
                    buffer = []

                    if src_format and target_format and src_format != target_format:
                        print(f"[UPSTREAM] Stream response transform enabled (no delay): {target_format} -> {src_format}")
                        combined_gen = self._create_combined_generator_with_transform(
                            buffer, aiter, response, target_format, src_format, use_gzip
                        )
                        # 转换后的流一定是 SSE
                        if pass_headers.get("content-type") != "text/event-stream":
                            pass_headers["content-type"] = "text/event-stream"
                    else:
                        combined_gen = self._create_combined_generator(buffer, aiter, response, use_gzip)
                        if pass_headers.get("content-type") != "text/event-stream":
                            pass_headers["content-type"] = "text/event-stream"

                    return StreamingResponse(combined_gen, headers=pass_headers)
            else:
                # 非流式请求（httpx 会自动处理 gzip 解压）
                response = await self.client.request(
                    method,
                    url,
                    headers=filtered_headers,
                    json=body
                )

                # 非 200：透传上游状态码 + 原始响应体（尽可能保留 content-type）
                if response.status_code != 200:
                    filtered_header_names = [
                        "content-length", "transfer-encoding", "content-encoding",
                        "set-cookie",
                        "strict-transport-security",
                        "content-security-policy", "content-security-policy-report-only",
                        "x-frame-options",
                        "x-content-type-options",
                        "x-xss-protection",
                        "permissions-policy",
                        "referrer-policy",
                    ]
                    pass_headers = {
                        k: v for k, v in response.headers.items()
                        if k.lower() not in filtered_header_names
                    }
                    media_type = response.headers.get("content-type")
                    return Response(
                        content=response.content,
                        status_code=response.status_code,
                        headers=pass_headers,
                        media_type=media_type,
                    )
                
                # 尝试解析 JSON
                try:
                    content = response.json()
                except Exception:
                    # 非 JSON 响应，返回文本
                    content = {"text": response.text, "status_code": response.status_code}
                
                # 如果需要响应转换
                if src_format and target_format and src_format != target_format:
                    try:
                        content = self._transform_response(content, target_format, src_format)
                    except Exception as e:
                        print(f"[ERROR] Response transform failed: {e}")
                        import traceback
                        traceback.print_exc()
                        # 转换失败时返回原始响应
                
                # 如果启用了内容检查（delay_stream_header 对非流式也生效）
                if delay_stream_header and response.status_code == 200:
                    # 如果进行了格式转换，应该用转换后的格式（src_format）进行检查
                    # 否则用上游格式（target_format）进行检查
                    if src_format and target_format and src_format != target_format:
                        check_format = src_format  # 使用转换后的格式
                    else:
                        check_format = target_format or "openai_chat"  # 使用上游格式
                    passed, error_msg = check_response_content(content, check_format)
                    
                    if not passed:
                        print(f"[CONTENT_CHECK_FAILED] {error_msg}")
                        return JSONResponse(
                            status_code=400,
                            content={
                                "error": {
                                    "code": "EMPTY_RESPONSE",
                                    "message": error_msg,
                                    "type": "content_validation_error"
                                }
                            }
                        )
                
                return JSONResponse(
                    status_code=response.status_code,
                    content=content
                )
        
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "UPSTREAM_ERROR",
                        "message": f"Upstream request failed: {str(e)}",
                        "type": "upstream_error"
                    }
                }
            )
    
    async def _create_combined_generator(
        self,
        buffer: list,
        aiter: AsyncIterator,
        response: httpx.Response,
        use_gzip: bool = False
    ) -> AsyncIterator[bytes]:
        """创建组合生成器（无转换）"""
        print(f"[UPSTREAM] STREAM Generator started (gzip={use_gzip})")
        
        if not use_gzip:
            # 不使用 gzip，直接透传
            try:
                # 先输出缓冲的内容
                print(f"[UPSTREAM] Yielding {len(buffer)} buffered chunks")
                for chunk in buffer:
                    yield chunk
                
                # 继续从迭代器读取剩余内容
                print(f"[UPSTREAM] Continuing with remaining stream...")
                try:
                    while True:
                        chunk = await aiter.__anext__()
                        yield chunk
                except StopAsyncIteration:
                    print(f"[UPSTREAM] Stream completed")
            except Exception as e:
                print(f"[UPSTREAM] ERROR Generator exception: {e}")
                import traceback
                traceback.print_exc()
                raise
            finally:
                print(f"[UPSTREAM] Closing response connection")
                await response.aclose()
        else:
            # 使用 gzip 压缩
            try:
                # 创建压缩对象
                import zlib
                compressor = zlib.compressobj(
                    level=6,
                    method=zlib.DEFLATED,
                    wbits=zlib.MAX_WBITS | 16  # 16 = gzip 格式
                )
                
                # 先压缩缓冲的内容
                print(f"[UPSTREAM] Compressing {len(buffer)} buffered chunks with gzip")
                for chunk in buffer:
                    compressed = compressor.compress(chunk)
                    if compressed:
                        yield compressed
                
                # 继续压缩剩余流
                print(f"[UPSTREAM] Continuing with remaining stream (compressed)...")
                try:
                    while True:
                        chunk = await aiter.__anext__()
                        compressed = compressor.compress(chunk)
                        if compressed:
                            yield compressed
                except StopAsyncIteration:
                    print(f"[UPSTREAM] Stream completed, flushing compressor")
                
                # 刷新压缩器，输出剩余数据
                final_compressed = compressor.flush()
                if final_compressed:
                    yield final_compressed
                    
            except Exception as e:
                print(f"[UPSTREAM] ERROR Generator exception: {e}")
                import traceback
                traceback.print_exc()
                raise
            finally:
                print(f"[UPSTREAM] Closing response connection")
                await response.aclose()
    
    async def _create_combined_generator_with_transform(
        self,
        buffer: list,
        aiter: AsyncIterator,
        response: httpx.Response,
        from_format: str,
        to_format: str,
        use_gzip: bool = False
    ) -> AsyncIterator[bytes]:
        """创建组合生成器（带格式转换）"""
        print(f"[UPSTREAM] STREAM Transform generator started: {from_format} -> {to_format} (gzip={use_gzip})")

        if use_gzip:
            print("[UPSTREAM] WARN gzip stream cannot be transformed; passthrough")
            async for chunk in self._create_combined_generator(buffer, aiter, response, use_gzip):
                yield chunk
            return

        transformer = create_stream_transformer(from_format, to_format)
        if transformer is None:
            print("[UPSTREAM] INFO no matching stream transformer; passthrough")
            async for chunk in self._create_combined_generator(buffer, aiter, response, use_gzip):
                yield chunk
            return

        try:
            for chunk in buffer:
                for out in transformer.feed(chunk):
                    yield out

            while True:
                try:
                    chunk = await aiter.__anext__()
                except StopAsyncIteration:
                    break
                for out in transformer.feed(chunk):
                    yield out
        finally:
            for tail in transformer.flush():
                yield tail
            print(f"[UPSTREAM] Closing response connection (transformed stream)")
            await response.aclose()
    
    def _transform_response(
        self,
        response: Dict[str, Any],
        from_format: str,
        to_format: str
    ) -> Dict[str, Any]:
        """
        转换上游响应格式
        
        Args:
            response: 上游响应（from_format 格式）
            from_format: 上游API格式
            to_format: 客户端期望格式
        """
        from ai_proxy.transform.formats.parser import get_parser
        
        # 获取解析器
        from_parser = get_parser(from_format)
        to_parser = get_parser(to_format)
        
        if not from_parser or not to_parser:
            print(f"[WARN] Parser not found: from={from_format}, to={to_format}")
            return response
        
        # 转换：上游格式 -> 内部格式 -> 客户端格式
        try:
            internal_resp = from_parser.resp_to_internal(response)
            client_resp = to_parser.internal_to_resp(internal_resp)
            print(f"[DEBUG] Response transformed: {from_format} -> {to_format}")
            return client_resp
        except Exception as e:
            print(f"[ERROR] Response transform exception: {e}")
            raise
