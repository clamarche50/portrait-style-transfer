from fastapi import APIRouter
from portrait_api.api.v1.routes import assets, health, jobs, styles

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(assets.router, prefix="/assets", tags=["assets"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(styles.router, prefix="/styles", tags=["styles"])
