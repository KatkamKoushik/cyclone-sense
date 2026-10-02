from celery import Celery
from backend.app.config import settings

celery_app = Celery(
    "cyclonesense_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_always_eager=settings.CELERY_TASK_ALWAYS_EAGER,  # Runs in-process when Redis broker is offline
    task_eager_propagates=True,
)


@celery_app.task(name="async_process_scientific_granule", bind=True)
def async_process_scientific_granule(self, filepath: str, source_origin: str):
    """
    Long-running asynchronous worker job:
    1. Inspects raw file
    2. Runs quality control checks across all channels
    3. Returns processing summary
    """
    from backend.app.scientific.reader import ScientificReader
    from backend.app.scientific.qc import QualityControlEngine

    meta = ScientificReader.inspect(filepath)
    channel_evals = {}
    for ch in meta.channels:
        ch_name = ch["name"]
        data, _ = ScientificReader.read_variable(filepath, ch_name)
        qc = QualityControlEngine.evaluate_array(data, ch_name)
        channel_evals[ch_name] = qc.to_dict()

    return {
        "status": "COMPLETED",
        "file_path": filepath,
        "format": meta.file_format,
        "sha256": meta.sha256_hash,
        "channel_evaluations": channel_evals,
    }
