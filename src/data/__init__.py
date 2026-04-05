"""Data module for dataset loading and preprocessing."""

from importlib import import_module

__all__ = [
    "BaseRadiologyDataset",
    "MIMICCXRDataset",
    "IUXRayDataset",
    "VQARADDataset",
    "SLAKEDataset",
    "MSCXRDataset",
    "VinDrCXRDataset",
    "PadChestDataset",
    "HFVQARADDataset",
    "HFIUXRayDataset",
]

_EXPORTS = {
    "BaseRadiologyDataset": ("src.data.base_dataset", "BaseRadiologyDataset"),
    "MIMICCXRDataset": ("src.data.mimic_cxr", "MIMICCXRDataset"),
    "IUXRayDataset": ("src.data.iu_xray", "IUXRayDataset"),
    "VQARADDataset": ("src.data.vqa_rad", "VQARADDataset"),
    "SLAKEDataset": ("src.data.slake", "SLAKEDataset"),
    "MSCXRDataset": ("src.data.ms_cxr", "MSCXRDataset"),
    "VinDrCXRDataset": ("src.data.vindr_cxr", "VinDrCXRDataset"),
    "PadChestDataset": ("src.data.padchest", "PadChestDataset"),
    "HFVQARADDataset": ("src.data.hf_vqa_rad", "HFVQARADDataset"),
    "HFIUXRayDataset": ("src.data.hf_iu_xray", "HFIUXRayDataset"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

