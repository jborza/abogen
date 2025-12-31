"""
TTS Configuration Management

This module manages TTS configuration including adapter selection, voice preferences,
and other TTS-related settings. It handles loading and saving configurations from/to
disk, validation, and provides default configurations.
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Optional, Any, List

from abogen.tts_adapters.registry import TTSAdapterRegistry

logger = logging.getLogger(__name__)


# Default language to voice mapping (Kokoro voices)
DEFAULT_VOICES = {
    "a": "af_alloy",      # American English - female
    "b": "bf_alice",      # British English - female
    "e": "ef_dora",       # Spanish - female
    "f": "ff_siwis",      # French - female
    "h": "hf_alpha",      # Hindi - female
    "i": "if_sara",       # Italian - female
    "j": "jf_alpha",      # Japanese - female
    "p": "pf_dora",       # Portuguese - female
    "z": "zf_xiaobei",    # Mandarin - female
}


@dataclass
class TTSConfig:
    """TTS configuration data structure"""
    
    # Active TTS adapter
    active_adapter: str = "kokoro"
    
    # Default speech speed
    default_speed: float = 1.0
    
    # Per-language voice preferences
    voice_preferences: Dict[str, str] = field(default_factory=lambda: DEFAULT_VOICES.copy())
    
    # Adapter-specific configurations
    adapter_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Feature flags
    disable_internet: bool = False
    use_gpu: bool = True
    cache_voices: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TTSConfig":
        """Create configuration from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def validate(self) -> tuple[bool, str]:
        """
        Validate the configuration.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if not self.active_adapter:
            return False, "active_adapter must not be empty"
        
        if not isinstance(self.default_speed, (int, float)) or self.default_speed <= 0:
            return False, "default_speed must be a positive number"
        
        if not isinstance(self.voice_preferences, dict):
            return False, "voice_preferences must be a dictionary"
        
        if not isinstance(self.adapter_configs, dict):
            return False, "adapter_configs must be a dictionary"
        
        return True, ""


class TTSConfigurationManager:
    """
    Manages TTS configuration loading, saving, and validation.
    
    This class handles:
    - Loading configuration from disk
    - Saving configuration to disk
    - Validating configuration against adapter requirements
    - Providing sensible defaults
    - Managing configuration per project or globally
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default location.
        """
        self._config: Optional[TTSConfig] = None
        self._config_path = config_path or self._get_default_config_path()
        self._registry = TTSAdapterRegistry.get_instance()
        self._logger = logging.getLogger(__name__)
    
    @staticmethod
    def _get_default_config_path() -> Path:
        """Get the default configuration file path"""
        from platformdirs import user_config_dir
        
        config_dir = Path(user_config_dir("abogen", "abogen"))
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "tts_config.json"
    
    def load_configuration(self) -> TTSConfig:
        """
        Load TTS configuration from disk.
        
        If the configuration file doesn't exist, returns default configuration.
        
        Returns:
            TTSConfig: The loaded configuration
        """
        if self._config is not None:
            return self._config
        
        try:
            if self._config_path.exists():
                with open(self._config_path, "r") as f:
                    data = json.load(f)
                    self._config = TTSConfig.from_dict(data)
                    self._logger.info(f"Loaded TTS configuration from {self._config_path}")
            else:
                self._config = TTSConfig()
                self._logger.info("Using default TTS configuration")
        except Exception as e:
            self._logger.warning(f"Failed to load configuration: {e}. Using defaults.")
            self._config = TTSConfig()
        
        return self._config
    
    def save_configuration(self, config: Optional[TTSConfig] = None) -> bool:
        """
        Save TTS configuration to disk.
        
        Args:
            config: Configuration to save. If None, saves the current configuration.
            
        Returns:
            bool: True if save was successful
        """
        if config is None:
            config = self._config or self.load_configuration()
        
        try:
            # Validate before saving
            is_valid, error_msg = config.validate()
            if not is_valid:
                self._logger.error(f"Configuration validation failed: {error_msg}")
                return False
            
            # Ensure directory exists
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self._config_path, "w") as f:
                json.dump(config.to_dict(), f, indent=2)
            
            self._config = config
            self._logger.info(f"Saved TTS configuration to {self._config_path}")
            return True
        
        except Exception as e:
            self._logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_configuration(self) -> TTSConfig:
        """
        Get the current TTS configuration.
        
        Returns:
            TTSConfig: The current configuration
        """
        if self._config is None:
            self.load_configuration()
        return self._config
    
    def update_configuration(self, **kwargs) -> bool:
        """
        Update configuration fields and save to disk.
        
        Args:
            **kwargs: Configuration fields to update
            
        Returns:
            bool: True if update was successful
        """
        config = self.get_configuration()
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                self._logger.warning(f"Unknown configuration field: {key}")
        
        # Save and return result
        return self.save_configuration(config)
    
    def get_adapter_config(self, adapter_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific adapter.
        
        Args:
            adapter_id: ID of the adapter
            
        Returns:
            dict: Adapter configuration
        """
        config = self.get_configuration()
        return config.adapter_configs.get(adapter_id, {})
    
    def set_adapter_config(self, adapter_id: str, config: Dict[str, Any]) -> bool:
        """
        Set configuration for a specific adapter.
        
        Args:
            adapter_id: ID of the adapter
            config: Configuration dictionary
            
        Returns:
            bool: True if successful
        """
        current_config = self.get_configuration()
        current_config.adapter_configs[adapter_id] = config
        return self.save_configuration(current_config)
    
    def get_voice_for_language(self, language_code: str) -> str:
        """
        Get the preferred voice for a language.
        
        Args:
            language_code: Language code (e.g., "a" for American English)
            
        Returns:
            str: Voice ID for the language
        """
        config = self.get_configuration()
        return config.voice_preferences.get(language_code, DEFAULT_VOICES.get(language_code, ""))
    
    def set_voice_for_language(self, language_code: str, voice_id: str) -> bool:
        """
        Set the preferred voice for a language.
        
        Args:
            language_code: Language code
            voice_id: Voice ID to use for this language
            
        Returns:
            bool: True if successful
        """
        config = self.get_configuration()
        config.voice_preferences[language_code] = voice_id
        return self.save_configuration(config)
    
    def reset_to_defaults(self) -> bool:
        """
        Reset configuration to default values.
        
        Returns:
            bool: True if successful
        """
        default_config = TTSConfig()
        return self.save_configuration(default_config)
    
    def validate_active_adapter(self) -> tuple[bool, str]:
        """
        Validate that the active adapter is available.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        config = self.get_configuration()
        available = self._registry.get_available_adapters()
        
        if config.active_adapter not in available:
            return False, f"Active adapter '{config.active_adapter}' is not available"
        
        return True, ""
    
    def get_default_device(self) -> str:
        """
        Get the default device based on configuration.
        
        Returns:
            str: Device name ("auto", "cpu", "cuda", or "mps")
        """
        import platform
        
        # Get device preference from config or determine automatically
        adapter_config = self.get_adapter_config(self.get_configuration().active_adapter)
        device = adapter_config.get("device", "auto")
        
        if device != "auto":
            return device
        
        # Auto-determine based on platform and hardware
        system = platform.system()
        
        # Check for Apple Silicon
        if system == "Darwin" and platform.processor() == "arm":
            return "mps"
        
        # Check for NVIDIA GPU
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        
        # Default to CPU
        return "cpu"
    
    def export_configuration(self, output_path: Path) -> bool:
        """
        Export configuration to a file.
        
        Args:
            output_path: Path to export to
            
        Returns:
            bool: True if successful
        """
        try:
            config = self.get_configuration()
            with open(output_path, "w") as f:
                json.dump(config.to_dict(), f, indent=2)
            self._logger.info(f"Exported configuration to {output_path}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_configuration(self, input_path: Path) -> bool:
        """
        Import configuration from a file.
        
        Args:
            input_path: Path to import from
            
        Returns:
            bool: True if successful
        """
        try:
            with open(input_path, "r") as f:
                data = json.load(f)
            
            config = TTSConfig.from_dict(data)
            is_valid, error_msg = config.validate()
            
            if not is_valid:
                self._logger.error(f"Imported configuration is invalid: {error_msg}")
                return False
            
            self.save_configuration(config)
            self._logger.info(f"Imported configuration from {input_path}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to import configuration: {e}")
            return False
    
    def print_configuration(self) -> None:
        """Print the current configuration to console (for debugging)"""
        config = self.get_configuration()
        print("\n=== TTS Configuration ===")
        print(f"Active Adapter: {config.active_adapter}")
        print(f"Default Speed: {config.default_speed}")
        print(f"Use GPU: {config.use_gpu}")
        print(f"Disable Internet: {config.disable_internet}")
        print(f"Cache Voices: {config.cache_voices}")
        print("\nVoice Preferences:")
        for lang, voice in config.voice_preferences.items():
            print(f"  {lang}: {voice}")
        print("\nAdapter Configurations:")
        for adapter_id, cfg in config.adapter_configs.items():
            print(f"  {adapter_id}: {cfg}")
        print("========================\n")


# Global configuration manager instance
_config_manager: Optional[TTSConfigurationManager] = None


def get_tts_config_manager() -> TTSConfigurationManager:
    """
    Get the global TTS configuration manager instance.
    
    Returns:
        TTSConfigurationManager: The global configuration manager
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = TTSConfigurationManager()
    return _config_manager


def get_tts_config() -> TTSConfig:
    """
    Get the current TTS configuration.
    
    Returns:
        TTSConfig: The current configuration
    """
    return get_tts_config_manager().get_configuration()


def save_tts_config() -> bool:
    """
    Save the current TTS configuration to disk.
    
    Returns:
        bool: True if successful
    """
    return get_tts_config_manager().save_configuration()


def reset_tts_config() -> bool:
    """
    Reset TTS configuration to default values.
    
    Returns:
        bool: True if successful
    """
    return get_tts_config_manager().reset_to_defaults()
