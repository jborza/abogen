"""
Coqui TTS Adapter

Provides integration with Coqui TTS for high-quality speech synthesis.
Supports multiple models including XTTS v2 for voice cloning and multilingual TTS.

Installation:
    pip install TTS

Models:
    - glow-tts: Fast, high-quality English TTS
    - xtts_v2: Multilingual TTS with voice cloning capability
    - tacotron2: Classic model with good quality

Configuration:
    {
        "model": "glow-tts" or "xtts_v2",  # Model to use
        "language": "en" or "multi",  # Language mode
        "speaker_wav": "/path/to/speaker.wav",  # Optional: for voice cloning with XTTS v2
        "device": "cuda" or "cpu",  # Device to use
    }
"""

import os
from typing import Dict, List, Any, Iterator, Optional
import io
import tempfile
import logging

from abogen.tts_adapters.base import (
    TTSAdapter,
    LocalTTSAdapter,
    Voice,
    AudioResult,
    TTSAdapterError,
    TTSAdapterInitError,
    TTSGenerationError,
)

logger = logging.getLogger(__name__)

# Coqui TTS available models
COQUI_MODELS = {
    "glow-tts": {
        "name": "Glow-TTS (English)",
        "languages": ["en"],
        "voice_cloning": False,
        "model_name": "tts_models/en/ljspeech/glow-tts",
        "description": "Fast, high-quality English TTS",
    },
    "tacotron2": {
        "name": "Tacotron2 (English)",
        "languages": ["en"],
        "voice_cloning": False,
        "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
        "description": "Classic model with excellent quality",
    },
    "xtts_v2": {
        "name": "XTTS v2 (Multilingual)",
        "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"],
        "voice_cloning": True,
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "description": "Multilingual TTS with voice cloning",
    },
}

# Coqui language codes mapping
COQUI_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "cs": "Czech",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "hu": "Hungarian",
    "ko": "Korean",
}


class CoquiTTSAdapter(LocalTTSAdapter):
    """
    Coqui TTS Adapter

    Uses Coqui TTS for high-quality speech synthesis.
    Supports multiple models including XTTS v2 for voice cloning and multilingual speech.
    """

    def __init__(self):
        """Initialize Coqui TTS adapter"""
        super().__init__()
        self._tts = None
        self._is_initialized = False
        self._config = {}
        self._model_name = "glow-tts"
        self._speaker_wav = None
        self._device = "cpu"

    @property
    def name(self) -> str:
        """Human-readable name of the adapter"""
        return "Coqui TTS"

    @property
    def adapter_type(self) -> str:
        """Type of adapter: 'local' or 'cloud'"""
        return "local"

    @property
    def adapter_id(self) -> str:
        """Unique identifier for this adapter"""
        return "coqui"

    def get_voices(self, language_code: Optional[str] = None) -> List[Voice]:
        """
        Get available voices for Coqui TTS.

        Coqui TTS has a single default voice per model, but XTTS v2 supports voice cloning.
        The language_code parameter filters voices by language.

        Args:
            language_code: Optional language code to filter voices

        Returns:
            List[Voice]: List of available voices
        """
        voices = []
        
        # Get current model info
        model_info = COQUI_MODELS.get(self._model_name, COQUI_MODELS["glow-tts"])
        supported_languages = model_info.get("languages", ["en"])
        
        # If voice cloning is available and speaker_wav is set, add cloned voice
        if model_info.get("voice_cloning") and self._speaker_wav:
            if language_code is None or language_code in supported_languages:
                voices.append(
                    Voice(
                        id="coqui_cloned",
                        name="Coqui Voice Clone",
                        language_code="a",  # Universal
                        gender="variable",
                        description="Voice cloned from speaker audio",
                    )
                )
        
        # Add default voices for each supported language
        for lang_code in supported_languages:
            if language_code is None or language_code == lang_code:
                voices.append(
                    Voice(
                        id=f"coqui_{lang_code}",
                        name=f"Coqui {COQUI_LANGUAGES.get(lang_code, lang_code)}",
                        language_code=lang_code,
                        gender="neutral",
                        description=f"Default voice for {COQUI_LANGUAGES.get(lang_code, lang_code)}",
                    )
                )
        
        return voices

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.

        Returns:
            List[str]: List of supported language codes
        """
        model_info = COQUI_MODELS.get(self._model_name, COQUI_MODELS["glow-tts"])
        return model_info.get("languages", ["en"])

    def get_configuration_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema for Coqui TTS adapter.

        Returns:
            Dict[str, Any]: Configuration schema with required and optional fields
        """
        return {
            "description": "Coqui TTS Configuration",
            "required_fields": {},
            "optional_fields": {
                "model": {
                    "type": "string",
                    "default": "glow-tts",
                    "enum": list(COQUI_MODELS.keys()),
                    "description": "Coqui TTS model to use",
                },
                "speaker_wav": {
                    "type": "string",
                    "default": "",
                    "description": "Path to speaker WAV file for voice cloning (XTTS v2 only)",
                },
                "device": {
                    "type": "string",
                    "default": "auto",
                    "enum": ["auto", "cpu", "cuda", "mps"],
                    "description": "Device to use for inference",
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
        model = config.get("model", "glow-tts").strip()
        if model not in COQUI_MODELS:
            available = ", ".join(COQUI_MODELS.keys())
            raise TTSAdapterError(
                f"Unknown model '{model}'. Available: {available}"
            )

        speaker_wav = config.get("speaker_wav", "").strip()
        if speaker_wav:
            model_info = COQUI_MODELS.get(model)
            if not model_info.get("voice_cloning"):
                raise TTSAdapterError(
                    f"Model '{model}' does not support voice cloning. "
                    f"Use 'xtts_v2' for voice cloning."
                )
            
            if not os.path.exists(speaker_wav):
                raise TTSAdapterError(f"Speaker WAV file not found: {speaker_wav}")

        return True

    def initialize(self, config: Dict[str, Any], device: str = "auto") -> None:
        """
        Initialize Coqui TTS adapter with configuration.

        Args:
            config: Configuration dictionary
            device: Device to use (auto, cpu, cuda, mps)

        Raises:
            TTSAdapterInitError: If initialization fails
        """
        if self._is_initialized:
            self.log_debug("Adapter already initialized, skipping")
            return

        try:
            self.log_info("Initializing Coqui TTS adapter...")

            # Validate configuration
            self.validate_configuration(config)

            # Get device
            if device == "auto":
                device = config.get("device", "auto")

            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._device = device
            self.log_debug(f"Using device: {device}")

            # Get model
            model_name = config.get("model", "glow-tts").strip()
            self._model_name = model_name
            model_info = COQUI_MODELS[model_name]

            # Import Coqui TTS
            try:
                from TTS.api import TTS
                self.log_debug("Coqui TTS imported successfully")
            except ImportError as e:
                raise TTSAdapterInitError(
                    f"Failed to import Coqui TTS: {e}. "
                    "Please install it with: pip install TTS"
                )

            # Initialize TTS model
            try:
                self.log_debug(f"Loading model: {model_info['model_name']}")
                self._tts = TTS(
                    model_name=model_info["model_name"],
                    gpu=(device == "cuda"),
                )
                self.log_debug(f"Model loaded successfully")
            except Exception as e:
                raise TTSAdapterInitError(
                    f"Failed to load Coqui TTS model '{model_name}': {str(e)}"
                )

            # Store speaker WAV if provided
            speaker_wav = config.get("speaker_wav", "").strip()
            if speaker_wav:
                self._speaker_wav = speaker_wav
                self.log_debug(f"Speaker WAV set: {speaker_wav}")

            self._is_initialized = True
            self._config = config
            self.log_info("Coqui TTS adapter initialized successfully")

        except TTSAdapterInitError:
            raise
        except Exception as e:
            error_msg = f"Failed to initialize Coqui TTS adapter: {str(e)}"
            self.log_error(error_msg)
            raise TTSAdapterInitError(error_msg)

    def generate_audio(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        split_pattern: Optional[str] = None,
        lang_code: Optional[str] = None,
        **kwargs,
    ) -> Iterator[AudioResult]:
        """
        Generate audio from text using Coqui TTS.

        Args:
            text: Text to synthesize
            voice: Voice ID (e.g., "coqui_en", "coqui_cloned")
            speed: Speech speed (1.0 = normal)
            split_pattern: Pattern for splitting text (optional)
            lang_code: Language code (required for non-cloned voices)
            **kwargs: Additional arguments (ignored)

        Yields:
            AudioResult: Audio chunks from Coqui TTS

        Raises:
            TTSGenerationError: If audio generation fails
        """
        if not self._is_initialized:
            raise TTSGenerationError("Adapter not initialized. Call initialize() first.")

        if not text or not text.strip():
            raise TTSGenerationError("Text cannot be empty")

        try:
            self.log_debug(f"Generating audio - Text length: {len(text)}, Voice: {voice}")

            # Determine language for non-cloned voices
            if voice.startswith("coqui_") and voice != "coqui_cloned":
                # Extract language code from voice ID (e.g., "coqui_en" -> "en")
                extracted_lang = voice.split("_", 1)[1]
                if extracted_lang != "cloned":
                    lang_code = extracted_lang
            
            if not lang_code:
                lang_code = "en"  # Default to English

            # Generate audio with Coqui TTS
            try:
                import numpy as np
                
                # Create temporary file for audio output
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                try:
                    # Only XTTS v2 is multilingual; glow-tts and tacotron2 are monolingual
                    is_multilingual = self._model_name == "xtts_v2"
                    
                    # Use voice cloning if available
                    if voice == "coqui_cloned" and self._speaker_wav:
                        self.log_debug(f"Generating cloned audio with speaker: {self._speaker_wav}")
                        kwargs = {
                            "text": text,
                            "file_path": tmp_path,
                            "speaker_wav": self._speaker_wav,
                        }
                        if is_multilingual:
                            kwargs["language"] = lang_code
                        self._tts.tts_to_file(**kwargs)
                    else:
                        # Regular synthesis
                        kwargs = {
                            "text": text,
                            "file_path": tmp_path,
                        }
                        if is_multilingual:
                            kwargs["language"] = lang_code
                        self._tts.tts_to_file(**kwargs)
                    
                    # Read generated audio
                    from scipy.io import wavfile
                    sample_rate, audio_data = wavfile.read(tmp_path)
                    
                    # Ensure audio is in float32 format
                    if audio_data.dtype != np.float32:
                        if audio_data.dtype == np.int16:
                            audio_data = audio_data.astype(np.float32) / 32768.0
                        else:
                            audio_data = audio_data.astype(np.float32)
                    
                    self.log_debug(f"Audio generated: {sample_rate}Hz, shape: {audio_data.shape}")
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

            except Exception as e:
                raise TTSGenerationError(f"Failed to generate audio with Coqui TTS: {str(e)}")

            # Calculate duration
            duration = len(audio_data) / sample_rate if isinstance(audio_data, np.ndarray) else 0

            # Create audio result
            audio_result = AudioResult(
                audio=audio_data,
                sample_rate=sample_rate,
                duration=duration,
                metadata={
                    "graphemes": text,
                    "adapter": "coqui",
                    "voice": voice,
                    "model": self._model_name,
                    "language": lang_code,
                },
            )

            self.log_debug(
                f"Generated audio: {sample_rate}Hz, ~{duration:.1f}s duration"
            )

            yield audio_result

        except TTSGenerationError:
            raise
        except Exception as e:
            error_msg = f"Failed to generate audio with Coqui TTS: {str(e)}"
            self.log_error(error_msg)
            raise TTSGenerationError(error_msg)

    def cleanup(self) -> None:
        """Clean up Coqui TTS adapter resources"""
        try:
            if self._tts is not None:
                self._tts = None
                self.log_debug("Coqui TTS cleaned up")
        except Exception as e:
            self.log_error(f"Error cleaning up Coqui TTS adapter: {e}")

    def get_adapter_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the adapter.

        Returns:
            dict: Adapter information including capabilities and metadata
        """
        model_info = COQUI_MODELS.get(self._model_name, COQUI_MODELS["glow-tts"])
        
        return {
            "id": self.adapter_id,
            "name": self.name,
            "type": self.adapter_type,
            "description": "Coqui TTS for high-quality multilingual speech synthesis",
            "voices": [v.id for v in self.get_voices()],
            "languages": self.get_supported_languages(),
            "features": {
                "voice_cloning": model_info.get("voice_cloning", False),
                "multilingual": len(model_info.get("languages", [])) > 1,
                "custom_voices": False,
                "speed_control": False,
                "emotion_control": False,
            },
            "models": {
                k: {
                    "name": v["name"],
                    "languages": v["languages"],
                    "voice_cloning": v["voice_cloning"],
                }
                for k, v in COQUI_MODELS.items()
            },
            "configuration": self.get_configuration_schema(),
        }
