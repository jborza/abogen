"""
OpenAI TTS Adapter

Provides integration with OpenAI's Text-to-Speech API.
Supports all OpenAI TTS models and voices with configurable options.

Configuration:
    {
        "api_key": "sk-...",  # OpenAI API key (required)
        "model": "tts-1" or "tts-1-hd",  # OpenAI model to use (optional, defaults to "tts-1-hd")
    }
"""

import os
from typing import Dict, List, Any, Iterator, Optional
from datetime import datetime
import io

from abogen.tts_adapters.base import (
    TTSAdapter,
    LocalTTSAdapter,
    CloudTTSAdapter,
    Voice,
    AudioResult,
    TTSAdapterError,
    TTSAdapterInitError,
    TTSGenerationError,
)

# OpenAI voice definitions
OPENAI_VOICES = {
    "alloy": {
        "name": "Alloy",
        "gender": "neutral",
        "language": "en",
        "description": "Neutral, balanced voice",
    },
    "ash": {
        "name": "Ash",
        "gender": "male",
        "language": "en",
        "description": "Smooth, calm male voice",
    },
    "coral": {
        "name": "Coral",
        "gender": "female",
        "language": "en",
        "description": "Warm, friendly female voice",
    },
    "echo": {
        "name": "Echo",
        "gender": "male",
        "language": "en",
        "description": "Warm, slightly deeper male voice",
    },
    "fable": {
        "name": "Fable",
        "gender": "male",
        "language": "en",
        "description": "Expressive, character-like male voice",
    },
    "onyx": {
        "name": "Onyx",
        "gender": "male",
        "language": "en",
        "description": "Deep, authoritative male voice",
    },
    "nova": {
        "name": "Nova",
        "gender": "female",
        "language": "en",
        "description": "Bright, energetic female voice",
    },
    "sage": {
        "name": "Sage",
        "gender": "neutral",
        "language": "en",
        "description": "Calm, measured voice with thoughtful tone",
    },
    "shimmer": {
        "name": "Shimmer",
        "gender": "female",
        "language": "en",
        "description": "Clear, crisp female voice",
    },
}

# OpenAI TTS models
OPENAI_MODELS = {
    "tts-1": {
        "name": "TTS 1 (Standard)",
        "latency": "low",
        "quality": "standard",
        "description": "Lowest latency model, good for real-time applications",
    },
    "tts-1-hd": {
        "name": "TTS 1 HD (High Definition)",
        "latency": "normal",
        "quality": "high",
        "description": "Higher quality audio, slightly higher latency",
    },
}

# Language mappings - OpenAI TTS supports all languages but we document the main ones
OPENAI_LANGUAGES = {
    "a": {"code": "en", "name": "English"},
    "b": {"code": "en", "name": "English"},
}


class OpenAIAdapter(CloudTTSAdapter):
    """
    OpenAI TTS Adapter

    Integrates with OpenAI's Text-to-Speech API for high-quality audio generation.
    Supports multiple voices and models with configurable options.
    """

    def __init__(self):
        """Initialize OpenAI adapter"""
        super().__init__()
        self._client = None
        self._model = "tts-1-hd"
        self._api_key = None

    @property
    def name(self) -> str:
        """Human-readable name of the adapter"""
        return "OpenAI TTS"

    @property
    def adapter_id(self) -> str:
        """Unique identifier for the adapter"""
        return "openai"

    @property
    def adapter_type(self) -> str:
        """Type of adapter: 'cloud'"""
        return "cloud"

    @property
    def api_endpoint(self) -> str:
        """OpenAI API endpoint"""
        return "https://api.openai.com/v1/audio/speech"

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.

        OpenAI TTS supports text in any language, but we document the main ones.

        Returns:
            List[str]: List of supported language codes
        """
        return list(OPENAI_LANGUAGES.keys())

    def get_voices(self, language_code: Optional[str] = None) -> List[Voice]:
        """
        Get available voices.

        OpenAI TTS has 6 voices available regardless of language.
        The language_code parameter is accepted for consistency but OpenAI voices work with all languages.

        Args:
            language_code: Optional language code filter (ignored - OpenAI voices work with all languages)

        Returns:
            List[Voice]: List of available voices
        """
        voices = []
        for voice_id, voice_info in OPENAI_VOICES.items():
            voices.append(
                Voice(
                    id=voice_id,
                    name=voice_info["name"],
                    language_code="en",  # OpenAI voices are primarily in English
                    gender=voice_info.get("gender"),
                    description=voice_info.get("description"),
                )
            )
        return voices

    def get_configuration_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema for OpenAI adapter.

        Returns:
            Dict[str, Any]: Configuration schema with required and optional fields
        """
        return {
            "description": "OpenAI Text-to-Speech API Configuration",
            "required_fields": {
                "api_key": {
                    "type": "string",
                    "description": "OpenAI API key (starts with 'sk-')",
                    "secret": True,
                },
            },
            "optional_fields": {
                "model": {
                    "type": "string",
                    "default": "tts-1-hd",
                    "enum": list(OPENAI_MODELS.keys()),
                    "description": "OpenAI TTS model to use",
                },
            },
        }

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Validate adapter configuration.

        Args:
            config: Configuration dictionary

        Returns:
            bool: True if configuration is valid

        Raises:
            TTSAdapterError: If configuration is invalid
        """
        # Check for required API key
        api_key = config.get("api_key", "").strip()
        if not api_key:
            raise TTSAdapterError(
                "API key is required for OpenAI adapter. "
                "Get one at https://platform.openai.com/api-keys"
            )

        if not api_key.startswith("sk-"):
            raise TTSAdapterError(
                "Invalid API key format. OpenAI API keys start with 'sk-'"
            )

        # Validate model choice
        model = config.get("model", "tts-1-hd")
        if model not in OPENAI_MODELS:
            raise TTSAdapterError(
                f"Invalid model: {model}. Must be one of: {', '.join(OPENAI_MODELS.keys())}"
            )

        return True

    def _validate_api_credentials(self, config: Dict[str, Any]) -> bool:
        """
        Validate API credentials for OpenAI.

        Args:
            config: Configuration containing API credentials

        Returns:
            bool: True if credentials are valid

        Raises:
            TTSAdapterError: If credentials are invalid
        """
        api_key = config.get("api_key", "").strip()
        
        if not api_key:
            raise TTSAdapterError("OpenAI API key is required")
        
        if not api_key.startswith("sk-"):
            raise TTSAdapterError("Invalid OpenAI API key format (must start with 'sk-')")
        
        return True

    def initialize(self, config: Dict[str, Any], device: str = "auto") -> None:
        """
        Initialize OpenAI adapter with configuration.

        Note: device parameter is ignored for cloud adapters as processing happens on OpenAI servers.

        Args:
            config: Configuration dictionary with api_key and optional model
            device: Device to use (ignored for cloud adapters)

        Raises:
            TTSAdapterInitError: If initialization fails
        """
        if self._is_initialized:
            self.log_debug("Adapter already initialized, skipping")
            return

        try:
            self.log_info("Initializing OpenAI TTS adapter...")

            # Validate configuration
            self.validate_configuration(config)

            # Get API key from config or environment
            api_key = config.get("api_key", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()

            if not api_key:
                raise TTSAdapterInitError(
                    "OpenAI API key not found in config or OPENAI_API_KEY environment variable"
                )

            # Get model preference
            self._model = config.get("model", "tts-1-hd")

            # Try to import and initialize OpenAI client
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=api_key)
                self._api_key = api_key
                self.log_debug("OpenAI client initialized successfully")
            except ImportError as e:
                raise TTSAdapterInitError(
                    f"Failed to import OpenAI client: {e}. "
                    "Please install it with: pip install openai"
                )

            # Test the API connection by making a simple request
            try:
                # Make a minimal request to verify API key is valid
                response = self._client.audio.speech.create(
                    model=self._model,
                    voice="alloy",
                    input="Test",  # Short test input
                    response_format="mp3",
                )
                self.log_debug(f"API connection successful, using model: {self._model}")
            except Exception as e:
                raise TTSAdapterInitError(
                    f"Failed to connect to OpenAI API: {str(e)}. "
                    "Please check your API key and internet connection."
                )

            self._is_initialized = True
            self._config = config
            self.log_info("OpenAI adapter initialized successfully")

        except TTSAdapterInitError:
            raise
        except Exception as e:
            error_msg = f"Failed to initialize OpenAI adapter: {str(e)}"
            self.log_error(error_msg)
            raise TTSAdapterInitError(error_msg)

    def generate_audio(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        split_pattern: Optional[str] = None,
        lang_code: Optional[str] = None,
        **kwargs
    ) -> Iterator[AudioResult]:
        """
        Generate audio from text using OpenAI TTS.

        OpenAI TTS returns complete audio files, so we yield one result per call.
        The speech_rate parameter in OpenAI (0.25-4.0) doesn't map exactly to our speed parameter.

        Args:
            text: Text to synthesize
            voice: OpenAI voice ID (alloy, echo, fable, onyx, nova, shimmer)
            speed: Speech speed (1.0 = normal; supported range: 0.5-2.0 approximately)
            split_pattern: Ignored for OpenAI (splitting done on client side)
            lang_code: Language code (OpenAI works with any language)
            **kwargs: Additional arguments (ignored)

        Yields:
            AudioResult: Audio chunks from OpenAI

        Raises:
            TTSGenerationError: If audio generation fails
        """
        if not self._is_initialized:
            raise TTSGenerationError(
                "Adapter not initialized. Call initialize() first."
            )

        if not text or not text.strip():
            raise TTSGenerationError("Text cannot be empty")

        if voice not in OPENAI_VOICES:
            available = ", ".join(OPENAI_VOICES.keys())
            raise TTSGenerationError(f"Unknown voice: {voice}. Available: {available}")

        try:
            self.log_debug(
                f"Generating audio - Text length: {len(text)}, Voice: {voice}, Speed: {speed}"
            )

            # OpenAI doesn't support speed directly, so we just log it
            if speed != 1.0:
                self.log_debug(
                    f"Speed parameter {speed} specified, but OpenAI API doesn't support custom speed. Using default speed."
                )

            # Call OpenAI API
            response = self._client.audio.speech.create(
                model=self._model,
                voice=voice,
                input=text,
                response_format="wav",  # Use WAV format for easier decoding
            )

            # Read the audio content from response
            audio_content = response.content

            # Convert WAV bytes to numpy array
            try:
                import numpy as np
                from scipy.io import wavfile
                import io

                # WAV is easier to decode than MP3
                sample_rate, audio_data = wavfile.read(io.BytesIO(audio_content))
                
                # Ensure it's in the right format
                if audio_data.dtype != np.int16:
                    audio_data = np.int16(audio_data)
                    
            except Exception as e:
                # If scipy fails, try librosa as fallback
                try:
                    import librosa
                    import io
                    audio_data, sample_rate = librosa.load(
                        io.BytesIO(audio_content), sr=None, mono=True
                    )
                    audio_data = np.int16(audio_data * 32767)
                except Exception as e2:
                    self.log_error(f"Error decoding audio: {e} (librosa fallback: {e2})")
                    raise TTSGenerationError(
                        f"Failed to decode audio from OpenAI. "
                        f"Please ensure scipy is installed: pip install scipy"
                    )

            # Create audio result
            audio_result = AudioResult(
                audio=audio_data,
                sample_rate=sample_rate,
                duration=len(audio_data) / sample_rate if hasattr(audio_data, "__len__") else 0,
                metadata={
                    "graphemes": text,
                    "adapter": "openai",
                    "voice": voice,
                    "model": self._model,
                },
            )

            self.log_debug(
                f"Generated audio: {sample_rate}Hz, "
                f"~{len(audio_data) / sample_rate:.1f}s duration"
            )

            yield audio_result

        except TTSGenerationError:
            raise
        except Exception as e:
            error_msg = f"Failed to generate audio with OpenAI: {str(e)}"
            self.log_error(error_msg)
            raise TTSGenerationError(error_msg)

    def cleanup(self) -> None:
        """Clean up OpenAI adapter resources"""
        try:
            # OpenAI client doesn't need explicit cleanup
            if self._client:
                self.log_debug("Cleaning up OpenAI adapter")
                self._client = None
                self._is_initialized = False
        except Exception as e:
            self.log_error(f"Error during cleanup: {e}")

    def get_adapter_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the adapter.

        Returns:
            Dict[str, Any]: Adapter information including supported models and voices
        """
        return {
            "name": self.name,
            "type": self.adapter_type,
            "models": list(OPENAI_MODELS.keys()),
            "voices": list(OPENAI_VOICES.keys()),
            "languages": self.get_supported_languages(),
            "features": {
                "voice_control": True,
                "speed_control": False,  # OpenAI doesn't support speed adjustment
                "voice_mixing": False,  # Cloud APIs typically don't support this
                "streaming": False,  # Requires separate API endpoint
            },
            "configuration": self.get_configuration_schema(),
        }


# Register the adapter
def register_openai_adapter():
    """Register OpenAI adapter with the registry"""
    from abogen.tts_adapters.registry import TTSAdapterRegistry

    registry = TTSAdapterRegistry.get_instance()
    registry.register_adapter("openai", OpenAIAdapter)
