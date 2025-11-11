# api_server.py - FastAPI Server for NLP to NoSQL with Enhanced Timeout Handling

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import asyncio
import logging
import time
from contextlib import asynccontextmanager
import threading
import signal
import sys

# Import our enhanced NLP processor
from No_Sql import processor, generate_mongo_query_with_timeout, run_mongo_query, test_connections

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global timeout configurations
DEFAULT_QUERY_TIMEOUT = 30  # seconds
DEFAULT_DB_TIMEOUT = 15    # seconds
MAX_TIMEOUT = 60          # maximum allowed timeout

# Lifespan manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting NLP-to-NoSQL API Server")
    try:
        connection_status = await asyncio.to_thread(test_connections, show_progress=True)
        if connection_status["overall"]:
            logger.info("✅ All systems ready")
        else:
            logger.warning("⚠️ Some connections failed - check configuration")
    except Exception as e:
        logger.error(f"❌ Startup test failed: {e}")
    
    yield
    
    logger.info("🛑 Shutting down NLP-to-NoSQL API Server")
    try:
        # Clean up resources
        processor.close_connections()
        logger.info("✅ Connections closed cleanly")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="NLP to NoSQL API",
    description="Convert natural language queries to MongoDB queries and execute them",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    input: str = Field(..., min_length=1)
    db: str = Field(default="testdb")
    collection: str = Field(default="testcoll")
    limit: Optional[int] = Field(default=50, ge=1, le=1000)
    timeout: Optional[int] = Field(default=DEFAULT_QUERY_TIMEOUT, ge=5, le=MAX_TIMEOUT, 
                                  description="Timeout in seconds for the entire operation")
    show_progress: Optional[bool] = Field(default=False, 
                                         description="Show progress information in logs")

class QueryResponse(BaseModel):
    ok: bool
    mongo_query: Dict[str, Any]
    total_matching: int
    results: list
    result_count: int
    execution_time: float
    query_generation_time: Optional[float] = None
    db_execution_time: Optional[float] = None
    timeout_used: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    mongodb_connected: bool
    llm_connected: bool
    timestamp: float
    version: str
    uptime: Optional[float] = None

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

# Track server start time for uptime calculation
SERVER_START_TIME = time.time()

# Timeout decorator
def with_timeout(timeout_seconds: int):
    """Decorator to add timeout to async functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        return wrapper
    return decorator

# Root endpoint
@app.get("/")
async def read_root():
    uptime = time.time() - SERVER_START_TIME
    return {
        "message": "NLP to NoSQL API is running",
        "version": "1.1.0",
        "status": "ok",
        "uptime": round(uptime, 2),
        "endpoints": {
            "health": "/health", 
            "query": "/query", 
            "query-with-progress": "/query?show_progress=true",
            "docs": "/docs", 
            "test": "/test"
        },
        "timeout_info": {
            "default_query_timeout": DEFAULT_QUERY_TIMEOUT,
            "default_db_timeout": DEFAULT_DB_TIMEOUT,
            "max_timeout": MAX_TIMEOUT
        }
    }

# Enhanced health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        connection_status = await asyncio.to_thread(test_connections, show_progress=False)
        uptime = time.time() - SERVER_START_TIME
        
        return HealthResponse(
            status="healthy" if connection_status["overall"] else "degraded",
            mongodb_connected=connection_status["mongodb"],
            llm_connected=connection_status["llm"],
            timestamp=time.time(),
            version="1.1.0",
            uptime=round(uptime, 2)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy", 
            mongodb_connected=False, 
            llm_connected=False,
            timestamp=time.time(), 
            version="1.1.0",
            uptime=round(time.time() - SERVER_START_TIME, 2)
        )

# Test endpoint with timeout
@app.get("/test")
@with_timeout(15)
async def test_endpoint():
    try:
        test_query = "Find all employees"
        start_time = time.time()
        
        result = await asyncio.to_thread(
            generate_mongo_query_with_timeout, 
            test_query, 
            timeout=10, 
            show_progress=False
        )
        
        execution_time = time.time() - start_time
        
        return {
            "status": "ok", 
            "test_query": test_query, 
            "generated_mongo": result,
            "execution_time": round(execution_time, 3)
        }
    except TimeoutError as e:
        logger.error(f"Test endpoint timed out: {e}")
        raise HTTPException(status_code=408, detail=str(e))
    except Exception as e:
        logger.error(f"Test endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

# Enhanced main query endpoint
@app.post("/query", response_model=QueryResponse)
async def process_query(req: QueryRequest):
    overall_start_time = time.time()
    query_gen_time = None
    db_exec_time = None
    
    try:
        logger.info(f"Processing query: '{req.input}' (timeout: {req.timeout}s)")
        
        # Step 1: Generate MongoDB query with timeout
        query_start_time = time.time()
        try:
            mongo_query = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_mongo_query_with_timeout,
                    req.input,
                    timeout=min(req.timeout - 5, 25),  # Leave 5 seconds buffer for DB query
                    show_progress=req.show_progress
                ),
                timeout=req.timeout * 0.7  # Use 70% of timeout for query generation
            )
            query_gen_time = time.time() - query_start_time
            logger.info(f"Query generation completed in {query_gen_time:.3f}s")
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Query generation timed out after {req.timeout * 0.7:.1f} seconds")
        
        # Step 2: Execute database query with remaining timeout
        remaining_timeout = req.timeout - query_gen_time - 1  # 1 second buffer
        if remaining_timeout < 2:
            remaining_timeout = 2  # Minimum timeout for DB operation
        
        db_start_time = time.time()
        try:
            total, results = await asyncio.wait_for(
                asyncio.to_thread(
                    run_mongo_query,
                    req.db,
                    req.collection,
                    mongo_query,
                    req.limit,
                    req.show_progress
                ),
                timeout=remaining_timeout
            )
            db_exec_time = time.time() - db_start_time
            logger.info(f"Database query completed in {db_exec_time:.3f}s")
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Database query timed out after {remaining_timeout:.1f} seconds")
        
        # Calculate total execution time
        total_execution_time = time.time() - overall_start_time
        
        logger.info(f"Query completed successfully in {total_execution_time:.3f}s total")
        
        return QueryResponse(
            ok=True,
            mongo_query=mongo_query,
            total_matching=total,
            results=results,
            result_count=len(results),
            execution_time=round(total_execution_time, 3),
            query_generation_time=round(query_gen_time, 3) if query_gen_time else None,
            db_execution_time=round(db_exec_time, 3) if db_exec_time else None,
            timeout_used=req.timeout
        )
        
    except TimeoutError as e:
        logger.error(f"Timeout error: {e}")
        raise HTTPException(status_code=408, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        total_execution_time = time.time() - overall_start_time
        logger.error(f"Unexpected error after {total_execution_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# New endpoint for long-running queries with progress tracking
@app.post("/query/async")
async def process_query_async(req: QueryRequest):
    """Async query endpoint that returns immediately with a task ID for polling"""
    # This could be implemented with a task queue like Celery
    # For now, we'll just return a placeholder
    return {
        "message": "Async queries not implemented yet",
        "suggestion": "Use the regular /query endpoint with appropriate timeout values"
    }

# Exception handlers
@app.exception_handler(TimeoutError)
async def timeout_exception_handler(request: Request, exc: TimeoutError):
    return JSONResponse(
        status_code=408,
        content={
            "ok": False,
            "error": str(exc),
            "error_type": "TIMEOUT_ERROR",
            "status_code": 408,
            "timestamp": time.time(),
            "suggestion": "Try increasing the timeout value or simplifying the query"
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "ok": False,
            "error": exc.detail,
            "error_type": "HTTP_ERROR",
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": "Internal server error",
            "error_type": "INTERNAL_ERROR",
            "status_code": 500,
            "timestamp": time.time(),
            "details": str(exc) if logger.level == logging.DEBUG else None
        }
    )

# Graceful shutdown handler
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}. Shutting down gracefully...")
    processor.close_connections()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app", 
        host="0.0.0.0", 
        port=8000, 
        log_level="info", 
        reload=True,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10
    )