import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    BigInteger,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ScientificProduct(Base):
    __tablename__ = "scientific_products"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_format: Mapped[str] = mapped_column(String(32), nullable=False)  # NETCDF4, HDF5, etc.
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    source_origin: Mapped[str] = mapped_column(String(128), nullable=False, index=True)  # NOAA_GOES, IBTRACS, etc.
    
    # Metadata extracted directly from scientific product attributes
    spatial_coverage: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    temporal_coverage: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    variables_manifest: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    channels_manifest: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    global_attributes: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    qc_records: Mapped[List["QualityControlRecord"]] = relationship(
        "QualityControlRecord", back_populates="product", cascade="all, delete-orphan"
    )
    storm_extractions: Mapped[List["StormExtraction"]] = relationship(
        "StormExtraction", back_populates="product", cascade="all, delete-orphan"
    )


class QualityControlRecord(Base):
    __tablename__ = "qc_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id: Mapped[str] = mapped_column(String(36), ForeignKey("scientific_products.id", ondelete="CASCADE"), nullable=False, index=True)
    
    qc_status: Mapped[str] = mapped_column(String(32), nullable=False)  # PASSED, DEGRADED, REJECTED
    physical_bounds_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    missing_pixel_percentage: Mapped[float] = mapped_column(Float, nullable=False)
    dqf_summary: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    anomalies: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    product: Mapped["ScientificProduct"] = relationship("ScientificProduct", back_populates="qc_records")


class StormExtraction(Base):
    __tablename__ = "storm_extractions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id: Mapped[str] = mapped_column(String(36), ForeignKey("scientific_products.id", ondelete="CASCADE"), nullable=False, index=True)
    
    storm_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    storm_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    center_latitude: Mapped[float] = mapped_column(Float, nullable=False)
    center_longitude: Mapped[float] = mapped_column(Float, nullable=False)
    radius_km: Mapped[float] = mapped_column(Float, nullable=False)
    
    tensor_shape: Mapped[List[int]] = mapped_column(JSON, nullable=False)
    channels_included: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    tensor_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    saved_tensor_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    extracted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    product: Mapped["ScientificProduct"] = relationship("ScientificProduct", back_populates="storm_extractions")


class ProvenanceRecord(Base):
    __tablename__ = "provenance_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)  # PRODUCT, TENSOR, INFERENCE
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    action: Mapped[str] = mapped_column(String(64), nullable=False)  # INGEST, QC, EXTRACTION, INFERENCE
    software_version: Mapped[str] = mapped_column(String(32), nullable=False)
    parameters: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    parent_provenance_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    storm_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    storm_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING", index=True)  # PENDING, PROCESSING, COMPLETED, FAILED
    
    input_parameters: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    
    result_intensity_kts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    result_category: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    result_category_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    result_probabilities: Mapped[Optional[List[float]]] = mapped_column(JSON, nullable=True)
    result_explainability: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    result_provenance_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class SatelliteObservationRecord(Base):
    __tablename__ = "satellite_observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    observation_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    platform: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    sensor: Mapped[str] = mapped_column(String(64), nullable=False)
    sensor_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)  # OPTICAL, SAR
    product: Mapped[str] = mapped_column(String(64), nullable=False)
    acquisition_time: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    bounds_json: Mapped[Dict[str, float]] = mapped_column(JSON, nullable=False, default=dict)
    spatial_resolution_meters: Mapped[float] = mapped_column(Float, nullable=False)
    bands_json: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    orbit_pass: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    cloud_coverage_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    file_granule_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    quality_status: Mapped[str] = mapped_column(String(32), nullable=False, default="PASSED")
    storage_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class ImpactAnalysisRecord(Base):
    __tablename__ = "impact_analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    cyclone_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    cyclone_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    location_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    
    target_latitude: Mapped[float] = mapped_column(Float, nullable=False)
    target_longitude: Mapped[float] = mapped_column(Float, nullable=False)
    
    pre_granule_id: Mapped[str] = mapped_column(String(255), nullable=False)
    post_granule_id: Mapped[str] = mapped_column(String(255), nullable=False)
    sensor_type: Mapped[str] = mapped_column(String(32), nullable=False)  # OPTICAL, SAR
    
    observed_severity: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence_level: Mapped[str] = mapped_column(String(32), nullable=False)
    
    total_area_km2: Mapped[float] = mapped_column(Float, nullable=False)
    affected_area_km2: Mapped[float] = mapped_column(Float, nullable=False)
    water_change_km2: Mapped[float] = mapped_column(Float, nullable=False)
    vegetation_change_km2: Mapped[float] = mapped_column(Float, nullable=False)
    surface_change_km2: Mapped[float] = mapped_column(Float, nullable=False)
    uncertain_area_km2: Mapped[float] = mapped_column(Float, nullable=False)
    
    cyclone_metrics_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    executive_summary: Mapped[Text] = mapped_column(Text, nullable=False)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class ChangeDetectionRecord(Base):
    __tablename__ = "change_detection_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id: Mapped[str] = mapped_column(String(128), ForeignKey("impact_analyses.analysis_id", ondelete="CASCADE"), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(64), nullable=False)
    statistics_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class NaturalLanguageQueryRecord(Base):
    __tablename__ = "nl_query_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    citations_json: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    uncertainty_level: Mapped[str] = mapped_column(String(32), nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


