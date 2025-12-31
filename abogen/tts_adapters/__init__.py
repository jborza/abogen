"""
TTS Adapters Package

This package provides an extensible adapter system for supporting multiple TTS engines.
Each adapter implements the TTSAdapter interface to provide a unified way of interacting
with different TTS services (local models or cloud APIs).

Basic Usage:
    from abogen.tts_adapters import TTSAdapter, Voice, AudioResult
    from abogen.tts_adapters.registry import TTSAdapterRegistry
    
    # Get the global registry
    registry = TTSAdapterRegistry.get_instance()
    
    # Initialize an adapter
    adapter = registry.get_adapter("kokoro", config={"device": "cuda"})
    
    # Get available voices
    voices = adapter.get_voices(language_code="a")
    
    # Generate audio
    for audio_result in adapter.generate_audio("Hello world", voice=voices[0].id):
        # Process audio
        pass
"""

from abogen.tts_adapters.base import (
    TTSAdapter,
    LocalTTSAdapter,
    CloudTTSAdapter,
    Voice,
    AudioResult,
    AdapterConfig,
    TTSAdapterError,
    TTSAdapterConfigError,
    TTSAdapterInitError,
    TTSGenerationError,
)
from abogen.tts_adapters.kokoro_adapter import KokoroAdapter
from abogen.tts_adapters.registry import (
    TTSAdapterRegistry,
    get_tts_adapter,
    get_available_tts_adapters,
    list_tts_adapters,
)

__all__ = [
    "TTSAdapter",
    "LocalTTSAdapter",
    "CloudTTSAdapter",
    "Voice",
    "AudioResult",
    "AdapterConfig",
    "TTSAdapterError",
    "TTSAdapterConfigError",
    "TTSAdapterInitError",
    "TTSGenerationError",
    "KokoroAdapter",
    "TTSAdapterRegistry",
    "get_tts_adapter",
    "get_available_tts_adapters",
    "list_tts_adapters",
]
