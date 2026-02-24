"""
FastAPI Backend Application - Main Entry Point

This is the main file that creates and configures the FastAPI application.

What is FastAPI?
- Modern Python web framework for building APIs
- Fast performance (comparable to NodeJS/Go)
- Automatic API documentation (Swagger UI)
- Built-in data validation with Pydantic
- Async support for high concurrency

Why FastAPI for this project?
- Easy to learn and use
- Excellent documentation
- Type hints throughout (better IDE support)
- Perfect for microservices architecture
- Active community and ecosystem

Application Structure:
1. Create FastAPI app instance
2. Configure CORS (allow frontend to call backend)
3. Add exception handlers (graceful error handling)
4. Include API routers (all endpoints)
5. Add startup events (initialization)
6. Run with Uvicorn server
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.core.settings import get_settings
from src.core.logging_setup import get_logger, setup_logging
from src.api import routes
from src.api import agent_routes

logger = get_logger(__name__)

# Get application settings
settings = get_settings()

# Configure root logging once so INFO/DEBUG logs are emitted consistently.
setup_logging(settings.log_level, log_file=settings.agent_log_file)


def create_app() -> FastAPI:
    """
    Create and Configure FastAPI Application
    
    This function creates the FastAPI application and configures:
    - Metadata (title, description, version)
    - CORS middleware (cross-origin requests)
    - Exception handlers (error responses)
    - API routers (endpoints)
    - Startup events (initialization)
    
    Returns:
        FastAPI: Configured application instance
    
    Design pattern:
    - Application factory pattern
    - Allows creating multiple app instances (useful for testing)
    - Centralizes configuration
    """
    
    # Create FastAPI application instance
    app = FastAPI(
        title="Bureaucracy Copilot Backend",
        description=(
            "FastAPI backend service for Bureaucracy Copilot. "
            "Provides RAG (Retrieval-Augmented Generation) capabilities "
            "for processing and querying German bureaucratic documents."
        ),
        version="2.0.0",
        # Auto-generate OpenAPI docs at /docs (Swagger UI) and /redoc (ReDoc)
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # ===== Configure CORS Middleware =====
    # CORS = Cross-Origin Resource Sharing
    # Browsers block requests from one origin (domain:port) to another by default
    # We need to explicitly allow frontend (port 8501) to call backend (port 8000)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins_list(),  # Allowed frontend URLs
        allow_credentials=True,  # Allow cookies/auth headers
        allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
        allow_headers=["*"],  # Allow all headers
    )
    
    # CORS example:
    # Frontend runs on: http://localhost:8501
    # Backend runs on: http://localhost:8000
    # Without CORS: Browser blocks frontend → backend requests
    # With CORS: Backend says "I allow requests from localhost:8501"
    
    # ===== Exception Handlers =====
    # Catch errors and return consistent JSON responses
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """
        Handle Pydantic Validation Errors
        
        When request data doesn't match expected schema, Pydantic raises
        RequestValidationError. This handler catches it and returns a
        user-friendly error response.
        
        Example:
        - Expected: {"text": "some string"}
        - Received: {"text": 123}  # Wrong type
        - Response: {"error": "text must be a string", "detail": [...]}
        """
        logger.warning(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Invalid request data",
                "detail": exc.errors()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """
        Handle All Other Unexpected Errors
        
        Catches any unhandled exceptions and returns a generic error response.
        Prevents server from crashing and leaking internal error details.
        
        In production:
        - Log full error with stack trace (for debugging)
        - Return generic message to user (security)
        """
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred"
            }
        )
    
    # ===== Include API Routers =====
    # Routers group related endpoints together
    # The routes.router already has /api prefix, so we don't add it here
    
    app.include_router(routes.router, tags=["API"])
    app.include_router(agent_routes.router, tags=["Agent"])
    
    # ===== Startup Events =====
    # Code to run when the application starts
    
    @app.on_event("startup")
    async def startup_event():
        """
        Application Startup Handler
        
        Runs once when the server starts.
        Good place for:
        - Database connections
        - Cache initialization
        - Pre-loading models
        - Validation checks
        """
        logger.info("🚀 Backend application starting up...")
        logger.info(f"📍 Backend URL: http://{settings.host}:{settings.port}")
        logger.info(f"📚 API Docs: http://localhost:{settings.port}/docs")
        logger.info(f"🔧 OpenAI configured: {settings.openai_api_key[:10]}..." if settings.openai_api_key else "❌ OpenAI API key not configured")
        logger.info(f"📊 ChromaDB path: {settings.chroma_persist_dir}")
        logger.info("✅ Backend ready to accept requests")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """
        Application Shutdown Handler
        
        Runs once when the server stops.
        Good place for:
        - Closing database connections
        - Saving state
        - Cleanup operations
        """
        logger.info("🛑 Backend application shutting down...")
    
    # ===== Root Endpoint =====
    # Simple endpoint at / for quick testing
    
    @app.get("/")
    async def root():
        """
        Root Endpoint
        
        Returns basic information about the API.
        Useful for quick health check.
        """
        return {
            "message": "Bureaucracy Copilot Backend API",
            "version": "2.0.0",
            "docs": f"http://localhost:{settings.port}/docs",
            "status": "running"
        }
    
    return app


# Create the application instance
# This is what Uvicorn will run
app = create_app()


# How to run this application:
#
# 1. Development mode (with auto-reload):
#    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
#
# 2. Production mode:
#    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
#
# 3. Using Python directly:
#    python -m uvicorn app.main:app --reload
#
# Command breakdown:
# - app.main:app = module path : variable name
# - --reload = restart on code changes (development only)
# - --host 0.0.0.0 = accept connections from any IP
# - --port 8000 = listen on port 8000
# - --workers 4 = run 4 worker processes (production)


# Understanding the request flow:
#
# 1. Frontend makes request: POST http://localhost:8000/api/ingest/text
# 2. Browser sends preflight CORS check (OPTIONS request)
# 3. Backend responds: "Yes, I allow requests from localhost:8501"
# 4. Browser sends actual POST request with JSON data
# 5. FastAPI receives request
# 6. CORS middleware checks origin (allowed)
# 7. Pydantic validates request data against TextIngestRequest model
#    - If invalid: validation_exception_handler returns error
#    - If valid: continue
# 8. Route handler (ingest_text function) executes:
#    - Validates API key
#    - Calls ingest_source service
#    - Returns IngestResponse
# 9. FastAPI serializes response to JSON
# 10. Backend sends JSON response to frontend
# 11. Frontend processes response


# Testing the API:
#
# 1. Interactive Swagger UI:
#    - Go to http://localhost:8000/docs
#    - Try out endpoints directly in browser
#
# 2. Using curl (command line):
#    curl -X POST "http://localhost:8000/api/ingest/text" \
#      -H "Content-Type: application/json" \
#      -d '{"text": "Test document", "source_name": "Test"}'
#
# 3. Using Python requests:
#    import requests
#    response = requests.post(
#        "http://localhost:8000/api/ingest/text",
#        json={"text": "Test document", "source_name": "Test"}
#    )
#    print(response.json())


# Deployment considerations:
#
# 1. Use environment variables for all secrets
# 2. Enable HTTPS in production (use nginx/caddy as reverse proxy)
# 3. Set proper CORS origins (not "*")
# 4. Add authentication/authorization
# 5. Use gunicorn or uvicorn with workers
# 6. Monitor with logging/metrics
# 7. Add rate limiting
# 8. Use connection pooling for databases
# 9. Enable compression
# 10. Set up health checks for load balancer
