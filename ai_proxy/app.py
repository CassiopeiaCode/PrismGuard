"""
PrismGuard (棱镜守卫) 主入口
FastAPI 应用启动文件
"""
import traceback
import sys
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from ai_proxy.proxy.router import router
from ai_proxy.config import settings
from ai_proxy.moderation.smart.scheduler import start_scheduler
from ai_proxy.utils.memory_guard import check_all_tracked, check_process_memory
from ai_proxy.utils.env_check import check_dependencies, DependencyError

app = FastAPI(
    title="PrismGuard",
    description="高级 AI API 中间件 - 智能审核 · 格式转换 · 透明代理",
    version="1.0.0"
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器 - 打印详细错误"""
    print(f"\n{'='*60}")
    print(f"[ERROR] Unhandled exception:")
    print(f"Path: {request.url.path}")
    print(f"Exception: {exc}")
    print(f"Traceback:")
    traceback.print_exc()
    print(f"{'='*60}\n")
    
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": type(exc).__name__}}
    )

# 全局标志：防止重复启动
_scheduler_started = False
_memory_guard_task = None

@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    global _scheduler_started, _memory_guard_task
    
    print("[INFO] PrismGuard 启动")

    # 执行环境依赖检查
    try:
        check_dependencies()
    except DependencyError as e:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[FATAL] 环境依赖检查失败，应用无法启动。", file=sys.stderr)
        print(f"错误: {e}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        # 在异步上下文中，不能直接 sys.exit()，可以通过抛出异常来阻止启动
        # 或者更直接地，对于 uvicorn，可以这样停止服务器
        # 这里我们选择更简单的方式，打印错误并让它继续，但理想情况下应该停止服务器
        # 对于生产环境，启动脚本应该处理非零退出码
        sys.exit(1) # 这在某些服务器上可能无法正常工作，但对于开发模式是有效的
    
    # 防止重复启动调度器（reload模式下可能多次触发）
    # Initialize RocksDB service early to avoid request-path opens and to elect the single writer worker.
    from ai_proxy.moderation.smart.sample_store_service import init_sample_store_service

    rocks_svc = await init_sample_store_service()

    # IMPORTANT (multi-worker): only the elected writer worker starts the scheduler,
    # otherwise multiple workers would schedule training concurrently.
    if rocks_svc.is_writer and not _scheduler_started:
        start_scheduler(check_interval_minutes=10)
        _scheduler_started = True
        print("[INFO] 模型训练调度器已启动 (writer worker)")
    elif not rocks_svc.is_writer:
        print("[INFO] 当前为非 writer worker：不启动训练调度器")
    
    # 启动内存守护后台任务（每30秒检查一次）
    _memory_guard_task = asyncio.create_task(memory_guard_loop())
    print("[INFO] 内存守护后台任务已启动")

async def memory_guard_loop():
    """内存守护后台循环 - 定期检查所有追踪的容器 + 进程总内存 + 释放空闲内存"""
    from ai_proxy.utils.memory_guard import periodic_memory_cleanup
    
    while True:
        try:
            await asyncio.sleep(30)  # 每30秒检查一次
            
            # 1. 检查容器内存
            cleared = check_all_tracked()
            if cleared > 0:
                print(f"[MEMORY_GUARD] 本次检查清空了 {cleared} 个超大容器")
            
            # 2. 检查进程总内存（兜底机制）
            check_process_memory()  # 如果超过2GB会自动退出
            
            # 3. 定期释放空闲内存给 OS（解决 glibc arena 不归还问题）
            periodic_memory_cleanup()
            
        except asyncio.CancelledError:
            print("[MEMORY_GUARD] 后台任务已取消")
            break
        except Exception as e:
            print(f"[MEMORY_GUARD] 后台任务异常: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global _memory_guard_task
    
    print("[INFO] PrismGuard 正在关闭...")
    
    # 取消内存守护任务
    if _memory_guard_task:
        _memory_guard_task.cancel()
        try:
            await _memory_guard_task
        except asyncio.CancelledError:
            pass
        print("[INFO] 内存守护任务已停止")
    
    # 清理 HTTP 客户端池
    from ai_proxy.proxy.upstream import cleanup_clients
    await cleanup_clients()
    print("[INFO] HTTP 客户端池已清理")
    
    # 清理 OpenAI 客户端
    from ai_proxy.moderation.smart.ai import cleanup_openai_clients
    await cleanup_openai_clients()
    print("[INFO] OpenAI 客户端池已清理")
    
    # 清理数据库连接池
    from ai_proxy.moderation.smart.sample_store_service import shutdown_sample_store_service
    await shutdown_sample_store_service()

    from ai_proxy.moderation.smart.storage import cleanup_pools
    cleanup_pools()
    print("[INFO] 数据库连接池已清理")
    
    # 清理关键词过滤器
    from ai_proxy.moderation.basic import cleanup_filters
    cleanup_filters()
    print("[INFO] 关键词过滤器已清理")
    
    print("[INFO] PrismGuard 已关闭")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ai_proxy.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
