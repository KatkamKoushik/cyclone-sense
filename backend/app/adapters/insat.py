from pathlib import Path
from typing import Any, Dict, Optional
import httpx
from backend.app.config import settings
from backend.app.adapters.base import (
    BaseSatelliteAdapter,
    AuthenticationConfigError,
    DataSourceUnavailableError,
)


class INSATAdapter(BaseSatelliteAdapter):
    """
    Adapter for ISRO MOSDAC INSAT-3D / INSAT-3DR meteorological satellite products (HDF5 format).
    Requires explicit authentication via ISRO MOSDAC API credentials.
    """

    MOSDAC_BASE_URL = "https://www.mosdac.gov.in/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(source_name="ISRO_INSAT")
        self.api_key = api_key or settings.ISRO_MOSDAC_API_KEY

    def check_configuration(self) -> Dict[str, Any]:
        has_key = bool(self.api_key and len(self.api_key.strip()) > 5)
        return {
            "source": self.source_name,
            "requires_auth": True,
            "configured": has_key,
            "auth_type": "MOSDAC_API_KEY",
            "instructions": (
                "To access authentic INSAT-3D/3DR HDF5 granules, register at https://www.mosdac.gov.in "
                "and set ISRO_MOSDAC_API_KEY in your .env or environment configuration."
            ) if not has_key else "Credentials configured successfully.",
        }

    async def check_connectivity(self) -> Dict[str, Any]:
        """
        Check genuine ISRO MOSDAC authentication status.
        Does not simulate or claim connected without genuine credentials.
        """
        if not self.api_key or len(self.api_key.strip()) <= 5:
            return {
                "connected": False,
                "status": "ACCESS_REQUIRED",
                "error": "ISRO MOSDAC credentials not configured (MOSDAC_API_KEY required).",
            }
        try:
            headers = {"Authorization": f"Bearer {self.api_key.strip()}"}
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.get(f"{self.MOSDAC_BASE_URL}/status", headers=headers)
                if resp.status_code == 200:
                    return {"connected": True, "status": "CONNECTED"}
                return {"connected": False, "status": "ACCESS_DENIED", "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"connected": False, "status": "DISCONNECTED", "error": str(e)}

    async def fetch_granule(
        self,
        product_identifier: str,
        destination_dir: Path,
    ) -> Path:
        """
        Fetch authentic INSAT-3D/3DR HDF5 granule from ISRO MOSDAC API.
        Never substitutes mock data if credentials are not configured.
        """
        if not self.api_key:
            raise AuthenticationConfigError(
                "ISRO MOSDAC requires authentication. Set ISRO_MOSDAC_API_KEY in environment or .env."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "CycloneSense-Scientific-Ingestion/0.1.0",
        }

        url = f"{self.MOSDAC_BASE_URL}/products/download/{product_identifier}"
        dest_file = destination_dir / f"{product_identifier}.h5"
        destination_dir.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.get(url, headers=headers)
                if response.status_code in [401, 403]:
                    raise AuthenticationConfigError(
                        f"MOSDAC rejected credentials for {product_identifier}: HTTP {response.status_code}"
                    )
                response.raise_for_status()
                with open(dest_file, "wb") as f:
                    f.write(response.content)
            except httpx.HTTPError as e:
                raise DataSourceUnavailableError(f"MOSDAC network request failed: {str(e)}") from e

        return dest_file
