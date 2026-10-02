import asyncio
from pathlib import Path
from backend.app.config import settings
from backend.app.db.session import init_db, async_session_factory
from backend.app.db.models import ScientificProduct, QualityControlRecord, ProvenanceRecord
from backend.app.scientific.reader import ScientificReader
from backend.app.scientific.qc import QualityControlEngine
from backend.app.scientific.provenance import ProvenanceTracker
from backend.app.scientific.grid_product_generator import create_reference_satellite_grid_netcdf
from sqlalchemy import select


async def seed():
    print("Initializing CycloneSense database schema...")
    await init_db()

    # 1. Ensure reference satellite grid product exists
    ref_grid_path = settings.DATA_RAW_DIR / "reference_satellite_grid.nc"
    if not ref_grid_path.exists():
        print(f"Generating reference NetCDF-4 multi-spectral satellite product at {ref_grid_path}...")
        create_reference_satellite_grid_netcdf(
            ref_grid_path, num_lats=200, num_lons=200, center_lat=18.5, center_lon=88.0
        )

    # 2. Ingest products if not already in DB
    files_to_ingest = [
        (ref_grid_path, "NOAA_GOES_LIKE_REFERENCE"),
        (settings.DATA_RAW_DIR / "IBTrACS.NI.v04r01.nc", "NOAA_NCEI_IBTRACS"),
    ]

    async with async_session_factory() as db:
        for file_path, origin in files_to_ingest:
            if not file_path.exists():
                print(f"Skipping {file_path} (does not exist)")
                continue

            meta = ScientificReader.inspect(file_path)
            stmt = select(ScientificProduct).where(ScientificProduct.sha256_hash == meta.sha256_hash)
            existing = (await db.execute(stmt)).scalars().first()
            if existing:
                print(f"Product already in database: {meta.filename} (ID: {existing.id})")
                continue

            # Evaluate QC on primary channel / variable
            primary_var = None
            if meta.channels:
                primary_var = meta.channels[0]["name"]
            elif meta.variables:
                for v in meta.variables:
                    if v["name"] not in ["lat", "lon", "time", "date_time", "charsn"]:
                        primary_var = v["name"]
                        break

            qc_res = None
            if primary_var:
                try:
                    data, _ = ScientificReader.read_variable(file_path, primary_var)
                    dqf_var = next((v["name"] for v in meta.variables if v.get("is_dqf")), None)
                    dqf_mask = None
                    if dqf_var:
                        dqf_mask, _ = ScientificReader.read_variable(file_path, dqf_var, apply_calibration=False)
                    qc_res = QualityControlEngine.evaluate_array(data, primary_var, dqf_mask=dqf_mask)
                except Exception as e:
                    print(f"QC eval error for {meta.filename}: {e}")

            if qc_res is None:
                import numpy as np
                qc_res = QualityControlEngine.evaluate_array(
                    data=np.array([1.0], dtype=np.float32), variable_name="generic_manifest"
                )

            product = ScientificProduct(
                filename=meta.filename,
                file_path=str(file_path),
                file_format=meta.file_format,
                file_size_bytes=meta.file_size_bytes,
                sha256_hash=meta.sha256_hash,
                source_origin=origin,
                spatial_coverage=meta.spatial_bounds,
                temporal_coverage=meta.temporal_bounds,
                variables_manifest=meta.variables,
                channels_manifest=meta.channels,
                global_attributes=meta.global_attributes,
            )
            db.add(product)
            await db.flush()

            qc_record = QualityControlRecord(
                product_id=product.id,
                qc_status=qc_res.status.value,
                physical_bounds_passed=qc_res.physical_bounds_passed,
                missing_pixel_percentage=qc_res.missing_pixel_percentage,
                dqf_summary=qc_res.dqf_summary,
                anomalies=qc_res.anomalies,
                metrics=qc_res.metrics,
            )
            db.add(qc_record)

            prov_dict = ProvenanceTracker.create_lineage_entry(
                entity_type="PRODUCT",
                entity_id=product.id,
                sha256_hash=product.sha256_hash,
                action="INGEST",
                software_version=settings.VERSION,
                parameters={
                    "source_origin": origin,
                    "file_format": meta.file_format,
                    "file_size": meta.file_size_bytes,
                    "dimensions": meta.dimensions,
                },
            )
            prov_record = ProvenanceRecord(
                id=prov_dict["id"],
                entity_type=prov_dict["entity_type"],
                entity_id=prov_dict["entity_id"],
                sha256_hash=prov_dict["sha256_hash"],
                action=prov_dict["action"],
                software_version=prov_dict["software_version"],
                parameters=prov_dict["parameters"],
                parent_provenance_id=None,
            )
            db.add(prov_record)
            await db.commit()
            print(f"Persisted product '{meta.filename}' (SHA256: {meta.sha256_hash[:16]}...) [QC: {qc_res.status.value}]")


if __name__ == "__main__":
    asyncio.run(seed())
