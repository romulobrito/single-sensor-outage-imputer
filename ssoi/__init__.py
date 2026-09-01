"""
Single-sensor outage imputer (SSOI).

Runtime package for target-conditional denoising regressor (TCDR) inference
under the deployment pattern: auxiliaries observed, target unavailable.
"""
from __future__ import annotations

from ssoi.bundle import ImputerBundle, load_bundle, save_bundle
from ssoi.model import TargetConditionalDenoisingRegressor
from ssoi.predict import VirtualSensor

__all__ = [
    "ImputerBundle",
    "TargetConditionalDenoisingRegressor",
    "VirtualSensor",
    "load_bundle",
    "save_bundle",
]

__version__ = "0.1.0"
