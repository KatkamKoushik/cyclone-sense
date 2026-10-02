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
