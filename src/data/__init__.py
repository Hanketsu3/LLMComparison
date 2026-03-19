"""
Data module for dataset loading and preprocessing.
"""

from src.data.base_dataset import BaseRadiologyDataset
from src.data.mimic_cxr import MIMICCXRDataset
from src.data.iu_xray import IUXRayDataset
from src.data.vqa_rad import VQARADDataset
from src.data.slake import SLAKEDataset
from src.data.ms_cxr import MSCXRDataset
from src.data.vindr_cxr import VinDrCXRDataset
from src.data.padchest import PadChestDataset

# HuggingFace auto-download datasets (no local setup needed)
from src.data.hf_vqa_rad import HFVQARADDataset
from src.data.hf_iu_xray import HFIUXRayDataset

__all__ = [
    "BaseRadiologyDataset",
    "MIMICCXRDataset",
    "IUXRayDataset",
    "VQARADDataset",
    "SLAKEDataset",
    "MSCXRDataset",
    "VinDrCXRDataset",
    "PadChestDataset",
    # HuggingFace
    "HFVQARADDataset",
    "HFIUXRayDataset",
]

