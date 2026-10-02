import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np


class ProvenanceTracker:
    """
    Immutable cryptographic provenance and scientific lineage tracker.
    Complies with W3C PROV standards.
    """

    @staticmethod
    def hash_file(filepath: Union[str, Path]) -> str:
        """Calculate SHA-256 digest of an on-disk scientific product."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def hash_array(array: np.ndarray) -> str:
        """Calculate SHA-256 digest of a numerical tensor (bytes + dtype + shape)."""
        hasher = hashlib.sha256()
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(str(array.shape).encode("utf-8"))
        # Use contiguous buffer for deterministic hashing
        hasher.update(np.ascontiguousarray(array).tobytes())
        return hasher.hexdigest()

    @classmethod
    def create_lineage_entry(
        cls,
        entity_type: str,
        entity_id: str,
        sha256_hash: str,
        action: str,
        software_version: str,
        parameters: Dict[str, Any],
        parent_provenance_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a serializable provenance lineage record.
        """
        record_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Sanitize parameters for JSON serialization
        def sanitize(obj):
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (datetime, Path)):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [sanitize(x) for x in obj]
            return obj

        clean_params = sanitize(parameters)

        return {
            "id": record_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "sha256_hash": sha256_hash,
            "action": action,
            "software_version": software_version,
            "parameters": clean_params,
            "parent_provenance_id": parent_provenance_id,
            "timestamp": timestamp,
        }

    @classmethod
    def create_lineage_record(
        cls,
        entity_type: str,
        entity_id: str,
        action: str,
        software_version: str,
        parameters: Dict[str, Any],
        sha256_hash: Optional[str] = None,
        data_hash: Optional[str] = None,
        parent_provenance_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience alias for create_lineage_entry with flexible hash parameter naming."""
        resolved_hash = sha256_hash or data_hash or ""
        return cls.create_lineage_entry(
            entity_type=entity_type,
            entity_id=entity_id,
            sha256_hash=resolved_hash,
            action=action,
            software_version=software_version,
            parameters=parameters,
            parent_provenance_id=parent_provenance_id,
        )
