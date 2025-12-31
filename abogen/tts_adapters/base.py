"""
Base TTS Adapter Interface and Data Structures

This module defines the abstract base class for all TTS adapters and common data structures.
All TTS adapters (Kokoro, OpenAI, AWS Polly, etc.) must inherit from TTSAdapter and implement
the required methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Iterator, Optional, Any, Type
import logging

logger = logging.getLogger(__name__)


class TTSAdapterError(Exception):
    """Base exception for TTS adapter errors"""
    pass


class TTSAdapterConfigError(TTSAdapterError):
    """Raised when adapter configuration is invalid"""
    pass


class TTSAdapterInitError(TTSAdapterError):
    """Raised when adapter initialization fails"""
    pass


class TTSGenerationError(TTSAdapterError):
    """Raised when audio generation fails"""
    pass


@dataclass
class Voice:
    """Represents a single voice available in a TTS adapter"""
    id: str
    name: str
    language_code: str
    gender: Optional[str] = None
    description: Optional[str] = None
    adapter_specific_data: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __hash__(self):
        return hash(f"{self.id}:{self.language_code}")

    def __eq__(self, other):
        if isinstance(other, Voice):
            return self.id == other.id and self.language_code == other.language_code
        return False


@dataclass
class AudioResult:
    """Result of TTS audio generation"""
    audio: Any  # numpy array or similar audio data
    sample_rate: int
    duration: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class AdapterConfig:
    """Configuration schema for an adapter"""
    name: str
    required_fields: List[str] = field(default_factory=list)
    optional_fields: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def get_schema(self) -> Dict[str, Any]:
        """Get the configuration schema as a dictionary"""
        return {
            'name': self.name,
            'required_fields': self.required_fields,
            'optional_fields': self.optional_fields,
            'description': self.description
        }


class TTSAdapter(ABC):
    """
    Abstract base class for all TTS adapters.
    
    All TTS adapters must inherit from this class and implement the required methods.
    Adapters can be either local (running on the user's machine) or cloud-based
    (making API calls to external services).
    """

    def __init__(self):
        """Initialize the adapter"""
        self._is_initialized = False
        self._config = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name of the adapter.
        
        Returns:
            str: Name like "Kokoro", "OpenAI TTS", "AWS Polly", etc.
        """
        pass

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """
        Unique identifier for the adapter (used internally).
        
        Returns:
            str: ID like "kokoro", "openai_tts", "aws_polly", etc.
        """
        pass

    @property
    @abstractmethod
    def adapter_type(self) -> str:
        """
        Type of adapter: 'local' or 'cloud'.
        
        Local adapters run models on the user's machine.
        Cloud adapters make API calls to external services.
        
        Returns:
            str: Either "local" or "cloud"
        """
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if adapter is initialized"""
        return self._is_initialized

    @abstractmethod
    def initialize(self, config: Dict[str, Any], device: str = "auto") -> None:
        """
        Initialize the adapter with the given configuration.
        
        Args:
            config: Configuration dictionary with adapter-specific settings
            device: Device to use ("cpu", "cuda", "mps", or "auto" for automatic selection)
            
        Raises:
            TTSAdapterInitError: If initialization fails
        """
        pass

    @abstractmethod
    def get_configuration_schema(self) -> AdapterConfig:
        """
        Get the configuration schema for this adapter.
        
        Returns:
            AdapterConfig: Configuration schema including required and optional fields
        """
        pass

    @abstractmethod
    def validate_configuration(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate adapter configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            tuple: (is_valid, error_message) where is_valid is True if config is valid
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List[str]: List of language codes like ["a", "b", "e", "j", etc.]
        """
        pass

    @abstractmethod
    def get_voices(self, language_code: Optional[str] = None) -> List[Voice]:
        """
        Get available voices, optionally filtered by language.
        
        Args:
            language_code: Optional language code to filter voices by.
                          If None, return all available voices.
        
        Returns:
            List[Voice]: List of available Voice objects
            
        Raises:
            TTSAdapterError: If voice retrieval fails
        """
        pass

    @abstractmethod
    def generate_audio(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        **kwargs
    ) -> Iterator[AudioResult]:
        """
        Generate audio from text using the specified voice.
        
        Args:
            text: Text to synthesize
            voice: Voice ID to use for synthesis
            speed: Speech speed multiplier (default 1.0)
            **kwargs: Additional adapter-specific parameters
            
        Yields:
            AudioResult: Audio chunks with metadata
            
        Raises:
            TTSGenerationError: If audio generation fails
            ValueError: If voice is not available
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up adapter resources.
        
        Called when the adapter is no longer needed or before shutdown.
        Adapters should use this to clean up GPU memory, close connections, etc.
        """
        pass

    def log_info(self, message: str) -> None:
        """Log an info message"""
        logger.info(f"[{self.name}] {message}")

    def log_warning(self, message: str) -> None:
        """Log a warning message"""
        logger.warning(f"[{self.name}] {message}")

    def log_error(self, message: str) -> None:
        """Log an error message"""
        logger.error(f"[{self.name}] {message}")

    def log_debug(self, message: str) -> None:
        """Log a debug message"""
        logger.debug(f"[{self.name}] {message}")


class LocalTTSAdapter(TTSAdapter):
    """
    Base class for local TTS adapters that run models on the user's machine.
    
    Subclasses should implement device-specific initialization and resource management.
    """

    @property
    def adapter_type(self) -> str:
        """Local adapters always return 'local'"""
        return "local"


class CloudTTSAdapter(TTSAdapter):
    """
    Base class for cloud-based TTS adapters that make API calls to external services.
    
    Subclasses should implement API authentication and request handling.
    """

    @property
    def adapter_type(self) -> str:
        """Cloud adapters always return 'cloud'"""
        return "cloud"

    @property
    @abstractmethod
    def api_endpoint(self) -> str:
        """Get the API endpoint URL"""
        pass

    @abstractmethod
    def _validate_api_credentials(self, config: Dict[str, Any]) -> bool:
        """
        Validate API credentials.
        
        Args:
            config: Configuration containing credentials
            
        Returns:
            bool: True if credentials are valid
        """
        pass
