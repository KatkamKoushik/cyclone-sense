from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from backend.app.config import settings
from backend.app.adapters.base import BaseSatelliteAdapter, DataSourceUnavailableError


class IBTrACSAdapter(BaseSatelliteAdapter):
    """
    Adapter for NOAA's International Best Track Archive for Climate Stewardship (IBTrACS).
    Authoritative public meteorological track NetCDF archive. No authentication required.
    """

    BASE_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf"

    # Known standard public NetCDF granules
    VALID_GRANULES = {
        "recent": "IBTrACS.last3years.v04r01.nc",
        "north_atlantic": "IBTrACS.NA.v04r01.nc",
        "north_indian": "IBTrACS.NI.v04r01.nc",
        "western_pacific": "IBTrACS.WP.v04r01.nc",
        "eastern_pacific": "IBTrACS.EP.v04r01.nc",
        "south_pacific": "IBTrACS.SP.v04r01.nc",
        "south_indian": "IBTrACS.SI.v04r01.nc",
    }

    def __init__(self):
        super().__init__(source_name="NOAA_IBTrACS")

    def check_configuration(self) -> Dict[str, Any]:
        """IBTrACS is open-access public data hosted by NOAA NCEI."""
        archive_path = settings.DATA_RAW_DIR / "IBTrACS.NI.v04r01.nc"
        return {
            "source": self.source_name,
            "requires_auth": False,
            "configured": True,
            "local_file_exists": archive_path.exists(),
            "base_url": self.BASE_URL,
            "available_subsets": list(self.VALID_GRANULES.keys()),
        }

    async def check_connectivity(self) -> Dict[str, Any]:
        """
        Verify local authoritative archive exists and is non-empty.
        """
        target = settings.DATA_RAW_DIR / "IBTrACS.NI.v04r01.nc"
        if target.exists() and target.stat().st_size > 1000:
            return {
                "connected": True,
                "status": "ARCHIVE_VERIFIED",
                "file_size_bytes": target.stat().st_size,
                "path": str(target),
            }
        return {
            "connected": False,
            "status": "ARCHIVE_MISSING",
            "error": "Authoritative IBTrACS.NI.v04r01.nc missing from data/raw directory.",
        }

    async def fetch_granule(
        self,
        granule_id: str,
        destination_dir: Path,
    ) -> Path:
        """
        Download authentic IBTrACS NetCDF product from NOAA NCEI servers.
        """
        filename = self.VALID_GRANULES.get(granule_id, granule_id)
        if not filename.endswith(".nc"):
            filename = f"{filename}.nc"

        url = f"{self.BASE_URL}/{filename}"
        dest_file = destination_dir / filename

        destination_dir.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            try:
                response = await client.get(url)
                if response.status_code == 404:
                    raise DataSourceUnavailableError(f"Granule {filename} not found on NOAA NCEI servers.")
                response.raise_for_status()
                with open(dest_file, "wb") as f:
                    f.write(response.content)
            except httpx.HTTPError as e:
                raise DataSourceUnavailableError(f"Failed to fetch IBTrACS product from NOAA: {str(e)}") from e

        return dest_file
