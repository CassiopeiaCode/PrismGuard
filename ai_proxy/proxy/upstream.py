"""
上游 HTTP 客户端封装 - 修复版
"""
import httpx
import json
from typing import Optional, Dict, Any
from fastapi.responses import StreamingResponse, JSONResponse
from ai_proxy.utils.memory_guard import check_container
from ai_proxy.proxy.stream_checker import StreamChecker, check_response_content


# 全局 HTTP 客户端池（每个 base_url 一个客户端）
_client_pool: Dict[str, httpx.AsyncClient] = {}


def get_or_create_client(base_url: str) -> httpx.AsyncClient:
    """获取或创建 HTTP 客户端（复用连接池）"""
    if base_url not in _client_pool:
        _client_pool[base_url] = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
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
        # 过滤掉不需要的头，并移除 Accept-Encoding 以避免 zstd 压缩
        filtered_headers = {
            k: v for k, v in headers.items()
            if k.lower() not in ["host", "content-length", "accept-encoding"]
        }
        # 明确请求 gzip 或不压缩（httpx 支持 gzip 自动解压）
        filtered_headers["Accept-Encoding"] = "gzip, deflate, identity"
        
        url = f"{self.base_url}{path}"
        
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
                
                if response.status_code != 200:
                    await response.aclose()
                    # 非 200，读取 body 并返回 JSONResponse
                    err_body = await response.aread()
                    try:
                        content = json.loads(err_body)
                    except:
                        content = {"error": err_body.decode('utf-8', errors='ignore')}
                    return JSONResponse(status_code=response.status_code, content=content)

                # 启用延迟检查
                if delay_stream_header:
                    checker = StreamChecker(target_format or "openai_chat")
                    buffer = []
                    valid = False
                    
                    try:
                        # 预读迭代器
                        aiter = response.aiter_bytes()
                        
                        # 预读循环
                        async for chunk in aiter:
                            buffer.append(chunk)
                            if checker.check_chunk(chunk):
                                valid = True
                                break
                            # 保护性限制：如果超过 8KB 还没满足条件，强制放行
                            if sum(len(b) for b in buffer) > 8192:
                                valid = True # 视为通过，避免一直卡住
                                break
                        
                        # 如果循环结束（流结束了）还没有 valid，检查一下
                        # 此时 valid 仍为 False，buffer 包含所有数据
                        if not valid:
                            # 检查是否真的为空或者只有空白字符
                            # 这里策略：如果流正常结束但内容很少，我们还是返回它
                            # 只有在发生异常断开时才认为是错误
                            raise Exception("Stream disconnected before valid content")
                            
                    except Exception as e:
                        print(f"[STREAM_PRE_READ_ERROR] {e}")
                        await response.aclose()
                        return JSONResponse(
                            status_code=502,
                            content={"error": {"code": "UPSTREAM_STREAM_ERROR", "message": "Stream disconnected before valid content"}}
                        )

                    # 构造新的生成器，先发 buffer，再发剩余流
                    async def combined_generator():
                        try:
                            for chunk in buffer:
                                yield chunk
                            async for chunk in aiter:
                                yield chunk
                        finally:
                            await response.aclose()

                    return StreamingResponse(
                        combined_generator(),
                        media_type="text/event-stream"
                    )
                
                else:
                    # 不延迟，直接透传
                    async def simple_generator():
                        try:
                            async for chunk in response.aiter_bytes():
                                yield chunk
                        finally:
                            await response.aclose()
                            
                    return StreamingResponse(
                        simple_generator(),
                        media_type="text/event-stream"
                    )
            else:
                # 非流式请求（httpx 会自动处理 gzip 解压）
                response = await self.client.request(
                    method,
                    url,
                    headers=filtered_headers,
                    json=body
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