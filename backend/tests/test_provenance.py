import numpy as np
import pytest
from backend.app.scientific.provenance import ProvenanceTracker


def test_hash_array_deterministic():
    arr1 = np.ones((3, 32, 32), dtype=np.float32)
    arr2 = np.ones((3, 32, 32), dtype=np.float32)
    arr3 = np.ones((3, 32, 32), dtype=np.float64)  # Different dtype

    hash1 = ProvenanceTracker.hash_array(arr1)
    hash2 = ProvenanceTracker.hash_array(arr2)
    hash3 = ProvenanceTracker.hash_array(arr3)

    assert hash1 == hash2
    assert len(hash1) == 64
    assert hash1 != hash3  # Different precision must have different cryptographic digest


def test_create_lineage_entry():
    entry = ProvenanceTracker.create_lineage_entry(
        entity_type="PRODUCT",
        entity_id="test-uuid-1234",
        sha256_hash="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        action="INGEST",
        software_version="0.1.0",
        parameters={"crop": [10, 20]},
    )
    assert entry["entity_type"] == "PRODUCT"
    assert entry["action"] == "INGEST"
    assert entry["software_version"] == "0.1.0"
    assert "timestamp" in entry
