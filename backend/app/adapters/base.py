from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx


class AuthenticationConfigError(Exception):
    """Raised when an external data source requires credentials that are not configured."""
    pass


class DataSourceUnavailableError(Exception):
    """Raised when an external data source fails or network is unreachable."""
    pass


class BaseSatelliteAdapter(ABC):
    """Abstract Base Class for real scientific satellite data providers."""

    def __init__(self, source_name: str):
        self.source_name = source_name

    @abstractmethod
    def check_configuration(self) -> Dict[str, Any]:
        """Verify whether required credentials and configurations are in place."""
        pass

    @abstractmethod
    async def fetch_granule(
        self,
        granule_id: str,
        destination_dir: Path,
    ) -> Path:
        """Fetch real scientific granule from provider."""
        pass
