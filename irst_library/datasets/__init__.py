"""
IRST detection datasets.
"""

from .sirst import SIRSTDataset, NUDTSIRSTDataset, IRSTD1KDataset
from .nuaa_sirst import NUAASIRSTDataset

__all__ = [
    "SIRSTDataset",
    "NUDTSIRSTDataset", 
    "IRSTD1KDataset",
    "NUAASIRSTDataset",
]
