"""
Kokoro TTS Adapter

Implements the TTSAdapter interface for the Kokoro text-to-speech model.
Kokoro is an open-weight TTS model with 82 million parameters that supports
multiple languages and provides high-quality speech synthesis.

Repository: https://github.com/hexgrad/kokoro
Model: https://huggingface.co/hexgrad/Kokoro-82M
"""

from typing import List, Dict, Iterator, Optional, Any, Tuple
import logging
import platform
import re

from abogen.tts_adapters.base import (
    LocalTTSAdapter,
    Voice,
    AudioResult,
    AdapterConfig,
    TTSAdapterInitError,
    TTSGenerationError,
)

logger = logging.getLogger(__name__)

# Language code mappings for Kokoro
# Format: language_code -> human_readable_name
KOKORO_LANGUAGES = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Brazilian Portuguese",
    "z": "Mandarin Chinese",
}

# Kokoro voices from the official model
KOKORO_VOICES = [
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
]

# Voice naming convention: [gender][language]_[name]
# gender: 'a'=female, 'm'=male
# language: 'a'=american english, 'b'=british english, etc.
GENDER_MAP = {
    "f": "female",
    "m": "male",
}


class KokoroAdapter(LocalTTSAdapter):
    """
    Kokoro TTS Adapter
    
    Provides text-to-speech synthesis using the Kokoro model.
    Supports voice mixing through weighted formulas.
    """

    def __init__(self):
        super().__init__()
        self._pipeline = None
        self._np_module = None
        self._repo_id = "hexgrad/Kokoro-82M"

    @property
    def name(self) -> str:
        return "Kokoro"

    @property
    def adapter_id(self) -> str:
        return "kokoro"

    def get_configuration_schema(self) -> AdapterConfig:
        """Get Kokoro configuration schema"""
        return AdapterConfig(
            name=self.adapter_id,
            required_fields=[],
            optional_fields={
                "device": {
                    "type": "string",
                    "default": "auto",
                    "description": "Device to use: 'cpu', 'cuda', 'mps', or 'auto'",
                    "choices": ["auto", "cpu", "cuda", "mps"],
                },
                "repo_id": {
                    "type": "string",
                    "default": "hexgrad/Kokoro-82M",
                    "description": "HuggingFace repository ID for the model",
                },
                "disable_internet": {
                    "type": "boolean",
                    "default": False,
                    "description": "Disable internet access for downloading models/voices",
                },
            },
            description="Kokoro TTS - A 82M parameter open-weight text-to-speech model",
        )

    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate Kokoro configuration"""
        if "device" in config:
            valid_devices = ["auto", "cpu", "cuda", "mps"]
            if config["device"] not in valid_devices:
                return False, f"Invalid device '{config['device']}'. Must be one of {valid_devices}"

        if "repo_id" in config and not isinstance(config["repo_id"], str):
            return False, "repo_id must be a string"

        if "disable_internet" in config and not isinstance(config["disable_internet"], bool):
            return False, "disable_internet must be a boolean"

        return True, ""

    def initialize(self, config: Dict[str, Any], device: str = "auto") -> None:
        """Initialize Kokoro pipeline"""
        if self._is_initialized:
            self.log_debug("Adapter already initialized, skipping")
            return

        try:
            self.log_info("Loading Kokoro model...")

            # Disable internet access if requested
            if config.get("disable_internet", False):
                import os
                os.environ["HF_HUB_OFFLINE"] = "1"
                self.log_info("Disabled internet access for HuggingFace Hub")

            # Load numpy and KPipeline
            try:
                import numpy as np
                from kokoro import KPipeline

                self._np_module = np
                self.log_debug("Successfully imported numpy and Kokoro")
            except ImportError as e:
                raise TTSAdapterInitError(
                    f"Failed to import Kokoro: {e}. Please install it with: pip install kokoro-tts"
                )

            # Determine device
            if device == "auto":
                # Check if device is specified in config, otherwise auto-detect
                device = config.get("device", None)
                if not device or device == "auto":
                    device = self._determine_device()
            self.log_info(f"Using device: {device}")

            # Initialize the pipeline
            repo_id = config.get("repo_id", self._repo_id)
            self._pipeline = KPipeline(
                lang_code="a",  # Default language, will be set per call
                repo_id=repo_id,
                device=device,
            )

            self._is_initialized = True
            self._config = config
            self.log_info("Kokoro adapter initialized successfully")

        except TTSAdapterInitError:
            raise
        except Exception as e:
            error_msg = f"Failed to initialize Kokoro adapter: {str(e)}"
            self.log_error(error_msg)
            raise TTSAdapterInitError(error_msg)

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return list(KOKORO_LANGUAGES.keys())

    def get_voices(self, language_code: Optional[str] = None) -> List[Voice]:
        """Get available voices, optionally filtered by language"""
        voices = []

        for voice_id in KOKORO_VOICES:
            # Parse voice ID: format is [language][gender]_[name]
            # Examples: af_alloy (american female), am_adam (american male), jf_alpha (japanese female)
            if len(voice_id) < 3:
                continue

            lang_code = voice_id[0]  # First character is language code
            gender_char = voice_id[1]  # Second character is gender code

            # Check if this voice matches the requested language
            if language_code is not None and lang_code != language_code:
                continue

            # Get human-readable values
            gender = GENDER_MAP.get(gender_char, "unknown")
            language_name = KOKORO_LANGUAGES.get(lang_code, "Unknown")

            voice = Voice(
                id=voice_id,
                name=voice_id.replace("_", " ").title(),
                language_code=lang_code,
                gender=gender,
                description=f"{gender.capitalize()} voice for {language_name}",
                adapter_specific_data={"kokoro_id": voice_id},
            )
            voices.append(voice)

        return voices

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
        Generate audio from text using Kokoro.

        Args:
            text: Text to synthesize
            voice: Voice ID or voice formula (e.g., "af_alloy" or "af_alloy*0.5+am_adam*0.5")
            speed: Speech speed multiplier (default 1.0)
            split_pattern: Regex pattern for text splitting (optional)
            lang_code: Language code for Kokoro (required for proper generation)
            **kwargs: Additional parameters (ignored)

        Yields:
            AudioResult: Audio chunks from Kokoro

        Raises:
            TTSGenerationError: If audio generation fails
        """
        if not self._is_initialized:
            raise TTSGenerationError("Adapter not initialized. Call initialize() first.")

        if not text or not text.strip():
            raise TTSGenerationError("Text cannot be empty")

        if not lang_code:
            raise TTSGenerationError("lang_code is required for Kokoro")

        try:
            self.log_debug(f"Generating audio for text: {text[:50]}...")
            self.log_debug(f"Voice: {voice}, Speed: {speed}, Lang: {lang_code}")

            # Set language for the pipeline
            self._pipeline.lang_code = lang_code

            # Handle voice formulas (weighted voice mixing) and voice tensors
            # Voice can be: string (voice ID), string with formula, or pre-loaded tensor
            if isinstance(voice, str):
                # Handle voice formulas (weighted voice mixing)
                if "*" in voice:
                    loaded_voice = self._load_voice_formula(voice)
                else:
                    # Validate and load voice
                    if voice not in KOKORO_VOICES:
                        raise TTSGenerationError(f"Unknown voice: {voice}")
                    loaded_voice = voice
            else:
                # Voice is already a tensor (pre-loaded weighted voice)
                loaded_voice = voice

            # Generate audio chunks
            for result in self._pipeline(
                text,
                voice=loaded_voice,
                speed=speed,
                split_pattern=split_pattern,
            ):
                # Extract audio data (handle both numpy arrays and torch tensors)
                audio_data = result.audio
                if hasattr(audio_data, "numpy"):
                    audio_data = audio_data.numpy()

                # Kokoro outputs at 24kHz
                audio_result = AudioResult(
                    audio=audio_data,
                    sample_rate=24000,
                    duration=len(audio_data) / 24000,
                    metadata={
                        "graphemes": getattr(result, "graphemes", ""),
                        "adapter": "kokoro",
                    },
                )
                yield audio_result

        except TTSGenerationError:
            raise
        except Exception as e:
            error_msg = f"Failed to generate audio with Kokoro: {str(e)}"
            self.log_error(error_msg)
            raise TTSGenerationError(error_msg)

    def load_single_voice(self, voice_id: str):
        """
        Load a single voice tensor for use in voice formula mixing.
        
        This method exposes the Kokoro pipeline's voice tensor loading capability
        for use in weighted voice mixing formulas.
        
        Args:
            voice_id: The voice ID (e.g., "af_alloy")
            
        Returns:
            Voice tensor (numpy array or torch tensor)
            
        Raises:
            TTSGenerationError: If voice loading fails
        """
        if not self._is_initialized:
            raise ValueError("Adapter not initialized. Call initialize() first.")
        
        if not voice_id:
            raise ValueError("voice_id cannot be empty")
        
        try:
            self.log_debug(f"Loading voice tensor for: {voice_id}")
            voice_tensor = self._pipeline.load_single_voice(voice_id)
            return voice_tensor
        except Exception as e:
            error_msg = f"Failed to load voice tensor for '{voice_id}': {str(e)}"
            self.log_error(error_msg)
            raise ValueError(error_msg)

    def cleanup(self) -> None:
        """Clean up Kokoro resources"""
        try:
            if self._pipeline is not None:
                # Clean up GPU memory if applicable
                if hasattr(self._pipeline, "model") and hasattr(self._pipeline.model, "cpu"):
                    self._pipeline.model.cpu()
                    self.log_debug("Moved model to CPU")

                # Clear any caches
                if self._np_module is not None:
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        self.log_debug("Cleared CUDA cache")
                    except Exception:
                        pass

                self._pipeline = None

            self._is_initialized = False
            self.log_info("Kokoro adapter cleaned up")
        except Exception as e:
            self.log_error(f"Error during cleanup: {str(e)}")

    # Private helper methods

    def _determine_device(self) -> str:
        """Determine the best device to use based on platform and available hardware"""
        system = platform.system()

        # Check for Apple Silicon
        if system == "Darwin" and platform.processor() == "arm":
            self.log_debug("Detected Apple Silicon, using MPS")
            return "mps"

        # Check for CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                self.log_debug("CUDA available, using cuda")
                return "cuda"
        except ImportError:
            pass

        # Default to CPU
        self.log_debug("Defaulting to CPU device")
        return "cpu"

    def _load_voice_formula(self, formula: str) -> Any:
        """
        Load a weighted voice formula and return the combined voice tensor.

        Formula format: "voice1*weight1+voice2*weight2+..."
        Example: "af_alloy*0.5+am_adam*0.5"

        Args:
            formula: Voice formula string

        Returns:
            Combined voice tensor

        Raises:
            TTSGenerationError: If formula parsing or voice loading fails
        """
        if not formula.strip():
            raise TTSGenerationError("Empty voice formula")

        try:
            # Calculate total weight for normalization
            total_weight = self._calculate_formula_weight(formula)
            if total_weight <= 0:
                raise TTSGenerationError("Total weight in voice formula must be positive")

            # Parse and load voices
            weighted_sum = None
            voices = formula.split("+")

            for term in voices:
                # Parse each term: "voice_name*weight"
                parts = term.strip().split("*")
                if len(parts) != 2:
                    raise TTSGenerationError(
                        f"Invalid voice term: '{term}'. Expected format: 'voice_name*weight'"
                    )

                voice_name = parts[0].strip()
                try:
                    weight = float(parts[1].strip())
                except ValueError:
                    raise TTSGenerationError(f"Invalid weight in term '{term}': must be a number")

                # Normalize weight
                weight /= total_weight

                # Validate voice exists
                if voice_name not in KOKORO_VOICES:
                    raise TTSGenerationError(f"Unknown voice in formula: {voice_name}")

                # Load voice tensor
                voice_tensor = self._pipeline.load_single_voice(voice_name)

                # Add to weighted sum
                if weighted_sum is None:
                    weighted_sum = weight * voice_tensor
                else:
                    weighted_sum += weight * voice_tensor

            if weighted_sum is None:
                raise TTSGenerationError("Failed to create weighted voice")

            self.log_debug(f"Loaded voice formula: {formula}")
            return weighted_sum

        except TTSGenerationError:
            raise
        except Exception as e:
            error_msg = f"Error parsing voice formula: {str(e)}"
            self.log_error(error_msg)
            raise TTSGenerationError(error_msg)

    @staticmethod
    def _calculate_formula_weight(formula: str) -> float:
        """Calculate the total weight in a voice formula"""
        weights = re.findall(r"\*\s*([\d.]+)", formula)
        total = sum(float(w) for w in weights)
        return total
