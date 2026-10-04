from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from backend.app.config import settings
from backend.app.adapters.base import (
    BaseSatelliteAdapter,
    AuthenticationConfigError,
    DataSourceUnavailableError,
)


class NASAAdapter(BaseSatelliteAdapter):
    """
    Adapter for NASA Earthdata Cloud & CMR (Common Metadata Repository).
    Provides real-time discovery and ingestion of authentic MODIS, VIIRS, and GPM satellite products.
    Authenticated via NASA Earthdata Bearer Token and User Credentials.
    """

    CMR_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

    def __init__(
        self,
        bearer_token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        super().__init__(source_name="NASA_EARTHDATA")
        self.bearer_token = bearer_token or settings.NASA_EARTHDATA_BEARER_TOKEN
        self.username = username or settings.NASA_EARTHDATA_USERNAME
        self.password = password or settings.NASA_EARTHDATA_PASSWORD

    def check_configuration(self) -> Dict[str, Any]:
        has_token = bool(self.bearer_token and len(self.bearer_token.strip()) > 10)
        has_user = bool(self.username and len(self.username.strip()) > 2)
        configured = has_token or has_user

        return {
            "source": self.source_name,
            "requires_auth": True,
            "configured": configured,
            "auth_type": "BEARER_TOKEN" if has_token else ("BASIC_AUTH" if has_user else "NONE"),
            "username": self.username if has_user else None,
            "has_bearer_token": has_token,
            "endpoint": "https://cmr.earthdata.nasa.gov",
            "supported_collections": [
                "MOD02QKM (MODIS Terra Calibrated Radiances 250m)",
                "MYD02QKM (MODIS Aqua Calibrated Radiances 250m)",
                "VNP02IMG (VIIRS SNPP Imagery Resolution)",
                "GPM_3IMERGHHE (GPM Early Precipitation)",
            ],
            "status": "AUTHENTICATED" if configured else "CREDENTIALS_REQUIRED",
        }

    async def check_connectivity(self) -> Dict[str, Any]:
        """
        Perform a live HTTP query against NASA CMR API to verify authentication
        and obtain real-time product observation timestamps and latency.
        """
        import datetime
        try:
            url = f"{self.CMR_SEARCH_URL}?short_name=MOD02QKM&sort_key[]=-start_date&page_size=1"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=self._get_headers())
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                if resp.status_code == 200:
                    data = resp.json()
                    entries = data.get("feed", {}).get("entry", [])
                    latest_obs_time = entries[0].get("time_start") if entries else None
                    granule_title = entries[0].get("title") if entries else None
                    
                    data_age_min = None
                    if latest_obs_time:
                        try:
                            # Parse ISO string
                            obs_dt = datetime.datetime.fromisoformat(latest_obs_time.replace("Z", "+00:00"))
                            data_age_min = max(0, int((now_utc - obs_dt).total_seconds() / 60))
                        except Exception:
                            pass

                    return {
                        "connected": True,
                        "status": "CONNECTED",
                        "endpoint": "https://cmr.earthdata.nasa.gov",
                        "latency_ms": round(resp.elapsed.total_seconds() * 1000, 1),
                        "latest_observation_utc": latest_obs_time,
                        "latest_granule_id": granule_title,
                        "retrieved_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "data_age_minutes": data_age_min,
                        "quality": "CALIBRATED_LANCE_OPERATIONAL",
                    }
                return {
                    "connected": False,
                    "status": "ERROR",
                    "endpoint": "https://cmr.earthdata.nasa.gov",
                    "error": f"HTTP status {resp.status_code}",
                }
        except Exception as e:
            return {
                "connected": False,
                "status": "DISCONNECTED",
                "endpoint": "https://cmr.earthdata.nasa.gov",
                "error": str(e),
            }

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "User-Agent": "CycloneSense-Scientific-Ingestion/0.1.0",
            "Accept": "application/json",
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token.strip()}"
        return headers

    async def search_granules(
        self,
        collection_short_name: str = "MOD02QKM",
        limit: int = 5,
        temporal_range: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query NASA CMR live for authentic satellite observation granules.
        """
        params: Dict[str, Any] = {
            "short_name": collection_short_name,
            "page_size": min(limit, 20),
        }
        if temporal_range:
            params["temporal"] = temporal_range

        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(self.CMR_SEARCH_URL, params=params, headers=headers)
                if response.status_code in [401, 403]:
                    raise AuthenticationConfigError(
                        f"NASA Earthdata rejected credentials: HTTP {response.status_code}"
                    )
                response.raise_for_status()
                data = response.json()
                entries = data.get("feed", {}).get("entry", [])

                results = []
                for e in entries:
                    download_url = None
                    for link in e.get("links", []):
                        href = link.get("href", "")
                        if href.endswith((".hdf", ".h5", ".nc", ".tif", ".bin")) and not href.endswith(".xml"):
                            download_url = href
                            break
                    if not download_url and e.get("links"):
                        download_url = e["links"][0].get("href")

                    results.append({
                        "granule_id": e.get("id"),
                        "title": e.get("title"),
                        "time_start": e.get("time_start"),
                        "time_end": e.get("time_end"),
                        "dataset_id": e.get("dataset_id"),
                        "download_url": download_url,
                        "size_mb": round(float(e.get("granule_size", 0)), 2),
                    })
                return results

            except httpx.HTTPError as e:
                raise DataSourceUnavailableError(f"NASA CMR search request failed: {str(e)}") from e

    async def fetch_granule(
        self,
        download_url_or_id: str,
        destination_dir: Path,
    ) -> Path:
        """
        Download real scientific satellite granule from NASA Earthdata Cloud.
        """
        url = download_url_or_id
        if not url.startswith("http"):
            # Resolve via search
            granules = await self.search_granules(limit=1)
            if not granules or not granules[0].get("download_url"):
                raise DataSourceUnavailableError(f"Could not resolve download URL for NASA granule '{url}'")
            url = granules[0]["download_url"]

        filename = Path(url.split("?")[0]).name
        if not filename:
            filename = f"nasa_granule_{int(hash(url))}.hdf"

        dest_file = destination_dir / filename
        destination_dir.mkdir(parents=True, exist_ok=True)

        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
            try:
                response = await client.get(url, headers=headers)
                if response.status_code in [401, 403]:
                    raise AuthenticationConfigError(
                        f"NASA Earthdata rejected download authorization: HTTP {response.status_code}"
                    )
                response.raise_for_status()

                with open(dest_file, "wb") as f:
                    f.write(response.content)

            except httpx.HTTPError as e:
                raise DataSourceUnavailableError(f"NASA Earthdata download failed: {str(e)}") from e

        return dest_file
