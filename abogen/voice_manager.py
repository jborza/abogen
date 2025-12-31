"""
Voice Management Helper Module

This module provides utility functions for managing voices across the application.
It bridges between the GUI components and the TTS adapter system, providing
a single source of truth for available voices.
"""

import logging
from typing import List, Optional, Dict, Any

from abogen.tts_adapters.registry import TTSAdapterRegistry
from abogen.tts_settings import get_tts_config_manager

logger = logging.getLogger(__name__)


class VoiceManager:
    """
    Manages voice operations across the application.
    
    This class provides a unified interface for:
    - Getting available voices from the active adapter
    - Validating voice IDs
    - Getting voice information
    """
    
    @staticmethod
    def get_available_voices(language_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available voices from the active TTS adapter.
        
        Args:
            language_code: Optional language code to filter voices
            
        Returns:
            List of voice dictionaries with id, name, language_code, gender, description
        """
        try:
            config_manager = get_tts_config_manager()
            tts_config = config_manager.get_configuration()
            
            # Get the active adapter
            registry = TTSAdapterRegistry.get_instance()
            
            # Get the saved config for this adapter
            adapter_config = tts_config.adapter_configs.get(tts_config.active_adapter, {})
            
            # Get adapter with saved config
            adapter = registry.get_adapter(tts_config.active_adapter, config=adapter_config)
            
            # Get voices from adapter
            voices = adapter.get_voices(language_code=language_code)
            
            # Convert Voice objects to dictionaries
            voice_list = []
            for voice in voices:
                voice_list.append({
                    'id': voice.id,
                    'name': voice.name,
                    'language_code': voice.language_code,
                    'gender': voice.gender,
                    'description': voice.description,
                })
            
            return voice_list
        except Exception as e:
            logger.error(f"Failed to get available voices: {e}")
            return []
    
    @staticmethod
    def get_voices_for_language(language_code: str) -> List[Dict[str, Any]]:
        """
        Get all voices for a specific language.
        
        Args:
            language_code: Language code (e.g., "a" for American English)
            
        Returns:
            List of voice dictionaries for the specified language
        """
        return VoiceManager.get_available_voices(language_code=language_code)
    
    @staticmethod
    def get_voice_ids() -> List[str]:
        """
        Get list of all available voice IDs.
        
        Returns:
            List of voice IDs
        """
        voices = VoiceManager.get_available_voices()
        return [v['id'] for v in voices]
    
    @staticmethod
    def is_valid_voice(voice_id: str) -> bool:
        """
        Check if a voice ID is valid for the active adapter.
        
        Args:
            voice_id: Voice ID to validate
            
        Returns:
            True if the voice ID is valid, False otherwise
        """
        available_voice_ids = VoiceManager.get_voice_ids()
        return voice_id in available_voice_ids
    
    @staticmethod
    def get_voice_info(voice_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific voice.
        
        Args:
            voice_id: Voice ID
            
        Returns:
            Voice information dictionary, or None if voice not found
        """
        voices = VoiceManager.get_available_voices()
        for voice in voices:
            if voice['id'] == voice_id:
                return voice
        return None
    
    @staticmethod
    def get_default_voice_for_language(language_code: str) -> Optional[str]:
        """
        Get the default/preferred voice for a language.
        
        Args:
            language_code: Language code
            
        Returns:
            Voice ID of the preferred voice, or None if not set
        """
        try:
            config_manager = get_tts_config_manager()
            return config_manager.get_voice_for_language(language_code)
        except Exception as e:
            logger.error(f"Failed to get default voice for {language_code}: {e}")
            return None
    
    @staticmethod
    def set_default_voice_for_language(language_code: str, voice_id: str) -> bool:
        """
        Set the preferred voice for a language.
        
        Args:
            language_code: Language code
            voice_id: Voice ID to set as preferred
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_manager = get_tts_config_manager()
            return config_manager.set_voice_for_language(language_code, voice_id)
        except Exception as e:
            logger.error(f"Failed to set default voice: {e}")
            return False
    
    @staticmethod
    def get_active_adapter_name() -> str:
        """
        Get the name of the currently active TTS adapter.
        
        Returns:
            Adapter name (e.g., "Kokoro")
        """
        try:
            config_manager = get_tts_config_manager()
            tts_config = config_manager.get_configuration()
            
            registry = TTSAdapterRegistry.get_instance()
            adapter = registry.get_adapter(tts_config.active_adapter)
            return adapter.name
        except Exception as e:
            logger.error(f"Failed to get active adapter name: {e}")
            return "Unknown"
    
    @staticmethod
    def get_active_adapter_id() -> str:
        """
        Get the ID of the currently active TTS adapter.
        
        Returns:
            Adapter ID (e.g., "kokoro")
        """
        try:
            config_manager = get_tts_config_manager()
            tts_config = config_manager.get_configuration()
            return tts_config.active_adapter
        except Exception as e:
            logger.error(f"Failed to get active adapter ID: {e}")
            return "unknown"


# Backward compatibility: Also export VOICES_INTERNAL as a list of voice IDs
def get_voices_internal() -> List[str]:
    """
    Get all voice IDs from the active adapter.
    
    This is for backward compatibility with code that uses VOICES_INTERNAL.
    
    Returns:
        List of all available voice IDs
    """
    return VoiceManager.get_voice_ids()


# Cache the voices list for performance
_voices_cache = None
_cache_adapter_id = None


def get_voices_internal_cached() -> List[str]:
    """
    Get all voice IDs from the active adapter with caching.
    
    Uses a cache that is invalidated when the active adapter changes.
    
    Returns:
        List of all available voice IDs
    """
    global _voices_cache, _cache_adapter_id
    
    try:
        current_adapter_id = VoiceManager.get_active_adapter_id()
        
        # If adapter changed or cache is empty, refresh
        if current_adapter_id != _cache_adapter_id or _voices_cache is None:
            _voices_cache = VoiceManager.get_voice_ids()
            _cache_adapter_id = current_adapter_id
        
        return _voices_cache
    except Exception as e:
        logger.error(f"Failed to get cached voices: {e}")
        return []


def clear_voices_cache():
    """Clear the voices cache"""
    global _voices_cache, _cache_adapter_id
    _voices_cache = None
    _cache_adapter_id = None
