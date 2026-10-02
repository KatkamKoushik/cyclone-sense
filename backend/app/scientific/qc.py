from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


class QCStatus(str, Enum):
    PASSED = "PASSED"
    DEGRADED = "DEGRADED"
    REJECTED = "REJECTED"


@dataclass
class QCResult:
    status: QCStatus
    physical_bounds_passed: bool
    missing_pixel_percentage: float
    dqf_summary: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "physical_bounds_passed": self.physical_bounds_passed,
            "missing_pixel_percentage": round(self.missing_pixel_percentage, 2),
            "dqf_summary": self.dqf_summary,
            "anomalies": self.anomalies,
            "metrics": self.metrics,
        }


class QualityControlEngine:
    """
    Scientific validation and physical sanity gate for meteorological satellite arrays.
    """

    DEFAULT_LIMITS = {
        "brightness_temperature": {"min": 160.0, "max": 340.0, "unit": "K"},
        "radiance": {"min": 0.0, "max": 250.0, "unit": "mW/(m2 sr cm-1)"},
        "wind_speed": {"min": 0.0, "max": 120.0, "unit": "m/s"},
        "central_pressure": {"min": 850.0, "max": 1050.0, "unit": "hPa"},
    }

    @classmethod
    def evaluate_array(
        cls,
        data: np.ndarray,
        variable_name: str,
        units: str = "",
        dqf_mask: Optional[np.ndarray] = None,
        max_missing_pct: float = 15.0,
    ) -> QCResult:
        """
        Evaluate an extracted numerical array against physical sanity laws and missing pixel limits.
        """
        total_elements = data.size
        if total_elements == 0:
            return QCResult(
                status=QCStatus.REJECTED,
                physical_bounds_passed=False,
                missing_pixel_percentage=100.0,
                anomalies=["Array has 0 elements"],
            )

        nan_mask = np.isnan(data)
        nan_count = int(np.sum(nan_mask))
        missing_pct = (nan_count / total_elements) * 100.0

        anomalies: List[str] = []
        bounds_passed = True
        status = QCStatus.PASSED

        valid_data = data[~nan_mask]

        metrics: Dict[str, Any] = {
            "total_pixels": total_elements,
            "valid_pixels": int(valid_data.size),
            "missing_pixels": nan_count,
            "missing_percentage": round(missing_pct, 2),
        }

        if valid_data.size > 0:
            data_min = float(np.min(valid_data))
            data_max = float(np.max(valid_data))
            data_mean = float(np.mean(valid_data))
            data_std = float(np.std(valid_data))

            metrics.update({
                "min": round(data_min, 3),
                "max": round(data_max, 3),
                "mean": round(data_mean, 3),
                "std": round(data_std, 3),
            })

            # Check variable physical limits
            norm_var = variable_name.lower()
            limit_key = None
            if any(term in norm_var for term in ["temp", "tb", "bt", "cmi", "band14", "band13", "band15"]):
                limit_key = "brightness_temperature"
            elif any(term in norm_var for term in ["rad", "radiance"]):
                limit_key = "radiance"
            elif any(term in norm_var for term in ["wind", "vmax", "speed"]):
                limit_key = "wind_speed"
            elif any(term in norm_var for term in ["pres", "pressure", "pmin"]):
                limit_key = "central_pressure"

            if limit_key and limit_key in cls.DEFAULT_LIMITS:
                limits = cls.DEFAULT_LIMITS[limit_key]
                p_min, p_max = limits["min"], limits["max"]

                if data_min < p_min:
                    bounds_passed = False
                    anomalies.append(f"Physical lower bound violated for {variable_name}: min {data_min:.2f} < {p_min} {limits['unit']}")
                if data_max > p_max:
                    bounds_passed = False
                    anomalies.append(f"Physical upper bound violated for {variable_name}: max {data_max:.2f} > {p_max} {limits['unit']}")
        else:
            bounds_passed = False
            anomalies.append(f"All {total_elements} pixels in {variable_name} are NaN/missing")

        # Missing pixel percentage check
        if missing_pct > max_missing_pct:
            if missing_pct >= 50.0:
                status = QCStatus.REJECTED
                anomalies.append(f"Excessive missing pixels: {missing_pct:.1f}% exceeds rejection limit (50%)")
            else:
                status = QCStatus.DEGRADED
                anomalies.append(f"Elevated missing pixels: {missing_pct:.1f}% exceeds nominal threshold ({max_missing_pct}%)")

        if not bounds_passed and status == QCStatus.PASSED:
            status = QCStatus.DEGRADED

        # DQF Evaluation if provided
        dqf_summary: Dict[str, Any] = {}
        if dqf_mask is not None and dqf_mask.size == total_elements:
            unique_flags, counts = np.unique(dqf_mask, return_counts=True)
            for flag, count in zip(unique_flags, counts):
                flag_pct = (int(count) / total_elements) * 100.0
                dqf_summary[f"flag_{int(flag)}"] = {
                    "count": int(count),
                    "percentage": round(flag_pct, 2)
                }
            # Check for non-nominal flags (flag > 0 in standard GOES ABI DQF)
            good_pixels = int(np.sum(dqf_mask == 0))
            good_pct = (good_pixels / total_elements) * 100.0
            dqf_summary["nominal_good_percentage"] = round(good_pct, 2)
            if good_pct < (100.0 - max_missing_pct) and status == QCStatus.PASSED:
                status = QCStatus.DEGRADED
                anomalies.append(f"Data Quality Flags indicate non-nominal instrument state (good pixels: {good_pct:.1f}%)")

        return QCResult(
            status=status,
            physical_bounds_passed=bounds_passed,
            missing_pixel_percentage=missing_pct,
            dqf_summary=dqf_summary,
            anomalies=anomalies,
            metrics=metrics,
        )
