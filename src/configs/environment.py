"""
Configuration Environment - Runtime preset and environment setup

Provides preset selection, validation, and merged config loading.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class RuntimePreset(Enum):
    """Available runtime presets."""
    SMOKE_CPU = "smoke_cpu"
    FREE_COLAB_T4 = "free_colab_t4"
    COLAB_PAID_MID = "colab_paid_mid"
    GPU_24G = "gpu_24g"
    HIGH_END_MULTI_GPU = "high_end_multi_gpu"


class EnvironmentProfile(Enum):
    """Available environment profiles."""
    BASE = "base"
    QWEN = "qwen"
    PHI = "phi"
    INTERNVL = "internvl"
    LLAMA = "llama"
    MEDICAL = "medical"
    SPECIALIST = "specialist"
    DEV = "dev"


@dataclass
class EnvironmentConfig:
    """Configuration for running experiments."""
    preset: RuntimePreset
    environment: EnvironmentProfile
    device: str  # "cpu" or "cuda"
    max_memory_gb: int
    batch_size: int
    image_size: int
    num_crops: int
    max_new_tokens: int
    quantization: Optional[str]  # "4bit", "8bit", or None
    
    # Feature flags
    use_flash_attention: bool = False
    use_vllm: bool = False
    enable_kv_cache: bool = True
    
    # Sampling
    num_samples: int = 100
    skip_heavy_metrics: bool = False
    
    def __str__(self) -> str:
        return (
            f"EnvironmentConfig("
            f"preset={self.preset.value}, "
            f"env={self.environment.value}, "
            f"device={self.device}, "
            f"max_mem={self.max_memory_gb}GB, "
            f"batch={self.batch_size}, "
            f"img_sz={self.image_size}, "
            f"quant={self.quantization})"
        )


class EnvironmentManager:
    """Manages environment configuration and validation."""
    
    def __init__(self, presets_file: Path = None, envs_dir: Path = None):
        """
        Initialize environment manager.
        
        Args:
            presets_file: Path to presets.yaml
            envs_dir: Path to envs/ directory
        """
        if presets_file is None:
            presets_file = Path(__file__).parent.parent.parent / "presets" / "presets.yaml"
        if envs_dir is None:
            envs_dir = Path(__file__).parent.parent.parent / "envs"
        
        self.presets_file = presets_file
        self.envs_dir = envs_dir
        
        # Load presets
        self.presets = self._load_presets()
        
        logger.info(f"Loaded {len(self.presets)} presets from {presets_file}")
    
    def _load_presets(self) -> Dict[str, Dict[str, Any]]:
        """Load presets from YAML."""
        if not self.presets_file.exists():
            raise FileNotFoundError(f"Presets file not found: {self.presets_file}")
        
        with open(self.presets_file, "r") as f:
            data = yaml.safe_load(f)
        
        return {k: v for k, v in data.items() if k != "presets"}
    
    def get_preset_config(self, preset: RuntimePreset) -> Dict[str, Any]:
        """Get raw preset configuration."""
        if preset.value not in self.presets:
            raise ValueError(f"Unknown preset: {preset.value}")
        return self.presets[preset.value]
    
    def build_environment_config(
        self,
        preset: RuntimePreset,
        environment: EnvironmentProfile = None,
        overrides: Dict[str, Any] = None,
    ) -> EnvironmentConfig:
        """
        Build environment configuration from preset.
        
        Args:
            preset: Runtime preset to use
            environment: Environment profile (auto-selected if None)
            overrides: Override specific settings
        
        Returns:
            EnvironmentConfig ready for use
        """
        preset_data = self.get_preset_config(preset)
        hw = preset_data.get("hardware", {})
        inf = preset_data.get("inference", {})
        eval_data = preset_data.get("evaluation", {})
        
        # Auto-select environment if not specified
        if environment is None:
            environment = self._infer_environment_profile(hw["device"])
        
        config = EnvironmentConfig(
            preset=preset,
            environment=environment,
            device=hw.get("device", "cpu"),
            max_memory_gb=hw.get("max_memory_gb", 8),
            batch_size=inf.get("batch_size", 1),
            image_size=inf.get("image_size", 224),
            num_crops=inf.get("num_crops", 1),
            max_new_tokens=inf.get("max_new_tokens", 256),
            quantization=inf.get("quantization"),
            use_flash_attention=inf.get("use_flash_attention", False),
            use_vllm=inf.get("use_vllm", False),
            enable_kv_cache=inf.get("enable_kv_cache", True),
            num_samples=eval_data.get("num_samples", 100),
            skip_heavy_metrics=eval_data.get("skip_heavy_metrics", False),
        )
        
        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown override key: {key}")
        
        return config
    
    def validate_config(self, config: EnvironmentConfig) -> bool:
        """Validate configuration for consistency."""
        # T4 + 24GB+: require quantization or skip 7B+ models
        if "t4" in config.device.lower() and config.max_memory_gb < 24:
            if config.quantization is None:
                logger.warning("T4 without quantization: may OOM on 7B+ models")
        
        # vLLM requires quantization=None (full precision)
        if config.use_vllm and config.quantization is not None:
            logger.warning("vLLM with quantization may not be supported")
        
        # Flash-attention requires GPU
        if config.use_flash_attention and config.device == "cpu":
            logger.warning("Flash-attention disabled for CPU inference")
            config.use_flash_attention = False
        
        return True
    
    def get_recommended_models_for_preset(self, preset: RuntimePreset) -> list:
        """Get list of recommended models for a preset."""
        preset_data = self.get_preset_config(preset)
        return preset_data.get("models", [])
    
    def get_environment_file(self, environment: EnvironmentProfile) -> Path:
        """Get path to environment YAML file."""
        env_file = self.envs_dir / f"{environment.value}.yaml"
        if not env_file.exists():
            raise FileNotFoundError(f"Environment file not found: {env_file}")
        return env_file
    
    def print_preset_summary(self, preset: RuntimePreset) -> None:
        """Print summary of a preset."""
        data = self.get_preset_config(preset)
        print(f"\n{'='*80}")
        print(f"🎯 PRESET: {data.get('name', preset.value)}")
        print(f"{'='*80}")
        
        hw = data.get("hardware", {})
        print(f"\n💻 Hardware:")
        print(f"  Device:        {hw.get('device')}")
        print(f"  GPU:           {hw.get('gpu_name', 'N/A')}")
        print(f"  VRAM:          {hw.get('vram_gb', 'N/A')}GB")
        print(f"  Precision:     {hw.get('precision', 'auto')}")
        
        inf = data.get("inference", {})
        print(f"\n⚙️  Inference:")
        print(f"  Batch Size:    {inf.get('batch_size', 1)}")
        print(f"  Image Size:    {inf.get('image_size', 224)}")
        print(f"  Max Tokens:    {inf.get('max_new_tokens', 256)}")
        print(f"  Quantization:  {inf.get('quantization', 'None')}")
        
        models = data.get("models", [])
        print(f"\n📦 Recommended Models ({len(models)}):")
        for model in models:
            print(f"  • {model}")
        
        warnings = data.get("warnings", [])
        if warnings:
            print(f"\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        
        print(f"\n{'='*80}\n")
    
    @staticmethod
    def _infer_environment_profile(device: str) -> EnvironmentProfile:
        """Infer best environment profile for device."""
        device_lower = device.lower()
        if "cpu" in device_lower:
            return EnvironmentProfile.BASE
        elif "t4" in device_lower or ("16" in device_lower and "gb" in device_lower):
            return EnvironmentProfile.QWEN  # Lightweight
        else:
            return EnvironmentProfile.SPECIALIST  # Full-featured


def get_default_config_for_preset(preset: RuntimePreset) -> EnvironmentConfig:
    """Quick helper to get default config for a preset."""
    manager = EnvironmentManager()
    return manager.build_environment_config(preset)
