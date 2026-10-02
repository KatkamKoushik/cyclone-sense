import os
import sys
from typing import Any, Dict
from fastapi import APIRouter
import redis
from backend.app.config import settings
from backend.app.db.session import engine
from backend.app.adapters.ibtracs import IBTrACSAdapter
from backend.app.adapters.goes import GOESAdapter
from backend.app.adapters.insat import INSATAdapter
from backend.app.ml.model import CycloneModelRegistry

router = APIRouter(prefix="/system", tags=["System & Infrastructure"])


@router.get("/health")
async def system_health() -> Dict[str, Any]:
    """
    Comprehensive system health and environment audit:
    - Database connectivity
    - Redis connectivity
    - Hardware / GPU acceleration
    - External satellite adapter configuration
    - Registered ML models
    """
    # 1. Database check
    db_status = "UNKNOWN"
    db_dialect = engine.url.get_backend_name()
    try:
        async with engine.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
            db_status = "HEALTHY"
    except Exception as e:
        db_status = f"ERROR: {str(e)}"

    # 2. Redis check
    redis_status = "DISCONNECTED"
    try:
        r = redis.from_url(settings.REDIS_URL, socket_timeout=1.0)
        if r.ping():
            redis_status = "HEALTHY"
    except Exception as e:
        redis_status = f"OFFLINE ({str(e)})"

    # 3. GPU / Hardware check
    gpu_info = {"available": False, "device_name": None}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
            }
    except ImportError:
        # Fallback to checking via environment / nvidia-smi if torch not installed
        pass

    # 4. External adapters status
    ibtracs_adapter = IBTrACSAdapter()
    goes_adapter = GOESAdapter()
    insat_adapter = INSATAdapter()

    # 5. Registered models
    models = [
        {
            "id": m.model_id,
            "version": m.version,
            "loaded": m.is_loaded,
            "description": m.description,
        }
        for m in CycloneModelRegistry.get_registered_models()
    ]

    return {
        "status": "OPERATIONAL" if db_status == "HEALTHY" else "DEGRADED",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "python_version": sys.version.split(" ")[0],
        "database": {
            "status": db_status,
            "backend": db_dialect,
            "url_masked": f"{db_dialect}://***",
        },
        "redis_task_queue": {
            "status": redis_status,
            "always_eager_fallback": settings.CELERY_TASK_ALWAYS_EAGER,
        },
        "compute": {
            "gpu": gpu_info,
            "platform": sys.platform,
        },
        "adapters": {
            "noaa_ibtracs": ibtracs_adapter.check_configuration(),
            "noaa_goes": goes_adapter.check_configuration(),
            "isro_insat": insat_adapter.check_configuration(),
        },
        "models": models,
    }
