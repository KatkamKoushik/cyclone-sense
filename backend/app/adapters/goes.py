from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from backend.app.adapters.base import BaseSatelliteAdapter, DataSourceUnavailableError


class GOESAdapter(BaseSatelliteAdapter):
    """
    Adapter for NOAA GOES-R Series ABI (Advanced Baseline Imager) NetCDF products.
    Accesses NOAA's Open Data Dissemination (NODD) on AWS Open Data.
    """

    BASE_S3_HTTP = "https://noaa-goes16.s3.amazonaws.com"

    def __init__(self, satellite: str = "goes16"):
        super().__init__(source_name=f"NOAA_{satellite.upper()}")
        self.satellite = satellite.lower()
        self.base_url = f"https://noaa-{self.satellite}.s3.amazonaws.com"

    def check_configuration(self) -> Dict[str, Any]:
        return {
            "source": self.source_name,
            "requires_auth": False,
            "configured": True,
            "endpoint": self.base_url,
            "supported_products": [
                "ABI-L2-CMIPF (Full Disk Cloud and Moisture Imagery)",
                "ABI-L2-CMIPC (CONUS)",
                "ABI-L2-CMIPM (Mesoscale)",
            ],
        }

    async def list_recent_granules(
        self,
        product: str = "ABI-L2-CMIPC",
        limit: int = 5,
        year: int = 2024,
        day_of_year: int = 270,
        hour: int = 18,
    ) -> List[Dict[str, Any]]:
        """
        List authentic NOAA GOES ABI granules live from NOAA AWS S3 public bucket.
        """
        import xml.etree.ElementTree as ET

        prefix = f"{product}/{year}/{day_of_year:03d}/{hour:02d}/"
        url = f"{self.base_url}/?list-type=2&prefix={prefix}&max-keys={min(limit, 20)}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                root = ET.fromstring(response.content)
                ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

                granules = []
                for elem in root.findall(".//s3:Contents", ns):
                    key_elem = elem.find("s3:Key", ns)
                    size_elem = elem.find("s3:Size", ns)
                    time_elem = elem.find("s3:LastModified", ns)
                    if key_elem is not None and key_elem.text and key_elem.text.endswith(".nc"):
                        key = key_elem.text
                        size_bytes = int(size_elem.text) if size_elem is not None and size_elem.text else 0
                        granules.append({
                            "key": key,
                            "filename": Path(key).name,
                            "download_url": f"{self.base_url}/{key}",
                            "size_bytes": size_bytes,
                            "size_mb": round(size_bytes / (1024 * 1024), 2),
                            "last_modified": time_elem.text if time_elem is not None else None,
                        })
                return granules
            except Exception as e:
                raise DataSourceUnavailableError(f"Failed to list GOES granules from NOAA S3: {str(e)}") from e

    async def fetch_granule(
        self,
        key_or_path: str,
        destination_dir: Path,
    ) -> Path:
        """
        Download authentic GOES ABI NetCDF product from NOAA S3 HTTP endpoint.
        Example key: 'ABI-L2-CMIPC/2024/270/18/OR_ABI-L2-CMIPC-M6C13_G16_s20242701801172_e20242701803545_c20242701804021.nc'
        """
        clean_key = key_or_path.lstrip("/")
        if clean_key.startswith("http"):
            url = clean_key
            clean_key = url.split(".com/")[-1]
        else:
            url = f"{self.base_url}/{clean_key}"

        filename = Path(clean_key).name
        dest_file = destination_dir / filename

        destination_dir.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            try:
                response = await client.get(url)
                if response.status_code == 404:
                    raise DataSourceUnavailableError(f"GOES product '{clean_key}' not found on NOAA bucket.")
                response.raise_for_status()
                with open(dest_file, "wb") as f:
                    f.write(response.content)
            except httpx.HTTPError as e:
                raise DataSourceUnavailableError(f"Failed to fetch GOES product: {str(e)}") from e

        return dest_file
