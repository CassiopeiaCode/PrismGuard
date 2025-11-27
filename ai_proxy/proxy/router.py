"""
主路由处理模块 - 支持多来源格式和工具调用
"""
import json
import urllib.parse
from typing import Optional, Tuple
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from ai_proxy.moderation.basic import basic_moderation
from ai_proxy.moderation.smart.ai import smart_moderation
from ai_proxy.transform.extractor import extract_text_from_internal
from ai_proxy.transform.formats.parser import detect_and_parse, get_parser
from ai_proxy.transform.formats.internal_models import InternalChatRequest
from ai_proxy.proxy.upstream import UpstreamClient

router = APIRouter()


def parse_url_config(cfg_and_upstream: str) -> Tuple[dict, str]:
    """
    解析 URL 中的配置和上游地址
    格式: {encoded_json_config}${upstream_url}
    """
    parts = cfg_and_upstream.split("$", 1)
    if len(parts) != 2:
        raise HTTPException(400, "Invalid URL format: expected {config}${upstream}")
    
    try:
        cfg_str = urllib.parse.unquote(parts[0])
        config = json.loads(cfg_str)
        upstream = parts[1]
        return config, upstream
    except Exception as e:
        raise HTTPException(400, f"Config parse error: {str(e)}")


async def process_request(
    config: dict,
    body: dict,
    path: str,
    headers: dict
) -> Tuple[bool, Optional[str], Optional[dict], Optional[str]]:
    """
    处理请求审核和格式转换
    
    Returns:
        (通过, 错误信息, 转换后的body或错误详情, 源格式名称)
    """
    try:
        return await _process_request_impl(config, body, path, headers)
    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print(f"[ERROR] Exception in process_request:")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        print(f"Traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise


async def _process_request_impl(
    config: dict,
    body: dict,
    path: str,
    headers: dict
) -> Tuple[bool, Optional[str], Optional[dict], Optional[str]]:
    """实际的请求处理逻辑"""
    print(f"\n[DEBUG] ========== 请求处理开始 ==========")
    print(f"  路径: {path}")
    
    # 格式转换配置
    transform_cfg = config.get("format_transform", {})
    transform_enabled = transform_cfg.get("enabled", False)
    
    print(f"  格式转换: {'启用' if transform_enabled else '禁用'}")
    
    if not transform_enabled:
        # 不转换，直接审核原始 body
        from ai_proxy.transform.extractor import extract_text_for_moderation
        text = extract_text_for_moderation(body, "openai_chat")
        
        print(f"  抽取文本长度: {len(text)} 字符")
        
        # 基础审核
        if config.get("basic_moderation", {}).get("enabled"):
            passed, reason = basic_moderation(text, config["basic_moderation"])
            if not passed:
                print(f"[DEBUG] ========== 请求被拒绝（基础审核） ==========\n")
                return False, reason, None, None
        
        # 智能审核
        if config.get("smart_moderation", {}).get("enabled"):
            passed, result = await smart_moderation(text, config["smart_moderation"])
            if not passed:
                print(f"[DEBUG] ========== 请求被拒绝（智能审核） ==========\n")
                # 构建详细错误信息
                details = {
                    "source": result.source,  # ai / bow_model / cache
                    "reason": result.reason,
                    "category": result.category,
                    "confidence": result.confidence
                }
                error_msg = f"Smart moderation blocked by {result.source}"
                if result.category:
                    error_msg += f" (category: {result.category})"
                if result.confidence is not None:
                    error_msg += f" (confidence: {result.confidence:.3f})"
                return False, error_msg, details, None
        
        print(f"[DEBUG] ========== 请求通过审核 ==========\n")
        return True, None, body, None
    
    # 检测并解析来源格式
    config_from = transform_cfg.get("from", "auto")
    src_format, internal_req = detect_and_parse(config_from, path, headers, body)
    
    if src_format is None:
        # 无法识别格式，透传
        print(f"[DEBUG] 无法识别格式，透传")
        print(f"[DEBUG] ========== 请求处理结束 ==========\n")
        return True, None, body, None
    
    print(f"  检测到格式: {src_format}")
    
    # 从内部格式抽取文本进行审核
    text = extract_text_from_internal(internal_req)
    print(f"  抽取文本长度: {len(text)} 字符")
    
    # 基础审核
    if config.get("basic_moderation", {}).get("enabled"):
        passed, reason = basic_moderation(text, config["basic_moderation"])
        if not passed:
            print(f"[DEBUG] ========== 请求被拒绝（基础审核） ==========\n")
            return False, reason, None, src_format
    
    # 智能审核
    if config.get("smart_moderation", {}).get("enabled"):
        passed, result = await smart_moderation(text, config["smart_moderation"])
        if not passed:
            print(f"[DEBUG] ========== 请求被拒绝（智能审核） ==========\n")
            # 构建详细错误信息
            details = {
                "source": result.source,
                "reason": result.reason,
                "category": result.category,
                "confidence": result.confidence
            }
            error_msg = f"Smart moderation blocked by {result.source}"
            if result.category:
                error_msg += f" (category: {result.category})"
            if result.confidence is not None:
                error_msg += f" (confidence: {result.confidence:.3f})"
            return False, error_msg, details, src_format
    
    # 格式转换
    target_format = transform_cfg.get("to", src_format)
    
    print(f"  目标格式: {target_format}")
    
    if target_format == src_format:
        # 目标格式与源格式相同，不转换
        print(f"  格式相同，无需转换")
        transformed_body = body
    else:
        # 转换到目标格式
        target_parser = get_parser(target_format)
        if target_parser is None:
            print(f"[DEBUG] 目标格式 {target_format} 不支持，使用源格式")
            transformed_body = body
        else:
            try:
                transformed_body = target_parser.to_format(internal_req)
                print(f"[DEBUG] 格式转换成功: {src_format} -> {target_format}")
            except Exception as e:
                print(f"[DEBUG] 格式转换失败: {e}")
                print(f"[DEBUG] ========== 请求处理失败 ==========\n")
                return False, f"Format transform error: {str(e)}", None, src_format
    
    print(f"[DEBUG] ========== 请求通过审核 ==========\n")
    return True, None, transformed_body, src_format


@router.api_route("/{cfg_and_upstream:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_entry(cfg_and_upstream: str, request: Request):
    """
    代理入口 - 支持多来源格式检测和转换
    
    URL 格式: /{url_encoded_config}${upstream_with_path}
    例如: /%7B...%7D$http://api.com/v1/chat/completions
    """
    # 解析配置，upstream_base 包含完整的上游 URL（含路径）
    try:
        config, upstream_full = parse_url_config(cfg_and_upstream)
        
        # 从 upstream_full 中分离 base_url 和 path
        from urllib.parse import urlparse
        parsed = urlparse(upstream_full)
        upstream_base = f"{parsed.scheme}://{parsed.netloc}"
        upstream_path = parsed.path or "/"
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "error": {
                    "code": "CONFIG_PARSE_ERROR",
                    "message": e.detail,
                    "type": "config_error"
                }
            }
        )
    
    # 获取请求体
    try:
        body = await request.json() if request.method in ["POST", "PUT"] else {}
    except:
        body = {}
    
    # 使用从 upstream_full 解析出的路径
    path = upstream_path
    
    # 处理审核和转换
    passed, error_msg, data, src_format = await process_request(
        config, body, path, dict(request.headers)
    )
    
    if not passed:
        error_response = {
            "code": "MODERATION_BLOCKED",
            "message": error_msg,
            "type": "moderation_error",
            "source_format": src_format
        }
        # 如果 data 是字典且包含审核详情，添加到响应中
        if isinstance(data, dict) and "source" in data:
            error_response["moderation_details"] = data
        
        return JSONResponse(
            status_code=400,
            content={"error": error_response}
        )
    
    # passed=True 时，data 是转换后的 body
    transformed_body = data
    
    # 转发到上游
    upstream_client = UpstreamClient(upstream_base)
    
    # 转发请求
    try:
        response = await upstream_client.forward_request(
            method=request.method,
            path=path,
            headers=dict(request.headers),
            body=transformed_body if transformed_body else body,
            is_stream=body.get("stream", False) if isinstance(body, dict) else False
        )
        return response
    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print(f"[ERROR] Proxy exception in router:")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        print(f"Traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "PROXY_ERROR",
                    "message": f"Proxy request failed: {str(e)}",
                    "type": "proxy_error"
                }
            }
        )