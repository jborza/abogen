"""
TTS Adapter Registry and Factory

This module provides a registry system for managing TTS adapters. It allows registration,
discovery, and initialization of adapters throughout the application.
"""

from typing import Dict, List, Optional, Type, Any
import logging

from abogen.tts_adapters.base import TTSAdapter, TTSAdapterInitError

logger = logging.getLogger(__name__)


class TTSAdapterRegistry:
    """
    Registry for managing TTS adapters.
    
    This is a singleton class that manages the registration, discovery, and initialization
    of all available TTS adapters. It supports both built-in adapters (Kokoro) and
    user-provided custom adapters.
    
    Usage:
        # Get the singleton instance
        registry = TTSAdapterRegistry.get_instance()
        
        # Get an initialized adapter
        adapter = registry.get_adapter("kokoro", config={"device": "cuda"})
        
        # List available adapters
        adapters = registry.get_available_adapters()
    """

    _instance = None
    _lock = None

    def __init__(self):
        """Initialize the registry"""
        self._adapters: Dict[str, Type[TTSAdapter]] = {}
        self._instances: Dict[str, TTSAdapter] = {}
        self._adapter_configs: Dict[str, Dict[str, Any]] = {}
        self._logger = logging.getLogger(__name__)

    @classmethod
    def get_instance(cls) -> "TTSAdapterRegistry":
        """
        Get the singleton instance of the registry.
        
        Returns:
            TTSAdapterRegistry: The singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._initialize_default_adapters()
        return cls._instance

    def register_adapter(
        self,
        adapter_id: str,
        adapter_class: Type[TTSAdapter],
        auto_initialize: bool = False,
    ) -> None:
        """
        Register an adapter class with the registry.
        
        Args:
            adapter_id: Unique identifier for the adapter (e.g., "kokoro", "openai_tts")
            adapter_class: The adapter class (must inherit from TTSAdapter)
            auto_initialize: If True, attempt to initialize the adapter on registration
            
        Raises:
            TypeError: If adapter_class doesn't inherit from TTSAdapter
            ValueError: If adapter_id is already registered
        """
        # Validate adapter class
        if not issubclass(adapter_class, TTSAdapter):
            raise TypeError(
                f"Adapter class {adapter_class.__name__} must inherit from TTSAdapter"
            )

        # Check for duplicates
        if adapter_id in self._adapters:
            self._logger.warning(
                f"Adapter '{adapter_id}' is already registered. Overwriting..."
            )

        # Register the adapter
        self._adapters[adapter_id] = adapter_class
        self._logger.info(f"Registered adapter: {adapter_id} -> {adapter_class.__name__}")

        # Auto-initialize if requested
        if auto_initialize:
            try:
                self.get_adapter(adapter_id)
                self._logger.info(f"Auto-initialized adapter: {adapter_id}")
            except Exception as e:
                self._logger.warning(f"Failed to auto-initialize adapter '{adapter_id}': {e}")

    def unregister_adapter(self, adapter_id: str) -> None:
        """
        Unregister an adapter from the registry.
        
        Args:
            adapter_id: Identifier of the adapter to unregister
        """
        if adapter_id in self._adapters:
            del self._adapters[adapter_id]
            self._logger.info(f"Unregistered adapter: {adapter_id}")

        # Clean up instances
        if adapter_id in self._instances:
            try:
                self._instances[adapter_id].cleanup()
            except Exception as e:
                self._logger.error(f"Error cleaning up adapter '{adapter_id}': {e}")
            del self._instances[adapter_id]

    def get_adapter(
        self,
        adapter_id: str,
        config: Optional[Dict[str, Any]] = None,
        force_reinit: bool = False,
    ) -> TTSAdapter:
        """
        Get an adapter instance, creating and initializing it if necessary.
        
        Args:
            adapter_id: Identifier of the adapter to get
            config: Configuration dictionary for the adapter. If None, uses cached config
            force_reinit: If True, reinitialize the adapter even if already initialized
            
        Returns:
            TTSAdapter: The adapter instance
            
        Raises:
            ValueError: If adapter_id is not registered
            TTSAdapterInitError: If adapter initialization fails
        """
        # Check if adapter is registered
        if adapter_id not in self._adapters:
            raise ValueError(f"Adapter '{adapter_id}' is not registered")

        # Return existing instance if available and not forcing reinitialization
        if adapter_id in self._instances and not force_reinit:
            return self._instances[adapter_id]

        # Create new instance
        try:
            adapter_class = self._adapters[adapter_id]
            adapter = adapter_class()

            # Get configuration
            if config is None:
                config = self._adapter_configs.get(adapter_id, {})
            else:
                # Cache the configuration
                self._adapter_configs[adapter_id] = config

            # Initialize the adapter
            adapter.initialize(config)

            # Cache the instance
            self._instances[adapter_id] = adapter
            self._logger.info(f"Initialized adapter: {adapter_id}")

            return adapter

        except TTSAdapterInitError:
            raise
        except Exception as e:
            error_msg = f"Failed to initialize adapter '{adapter_id}': {str(e)}"
            self._logger.error(error_msg)
            raise TTSAdapterInitError(error_msg)

    def get_available_adapters(self) -> List[str]:
        """
        Get list of registered adapter identifiers.
        
        Returns:
            List[str]: List of adapter IDs
        """
        return list(self._adapters.keys())

    def get_adapter_info(self, adapter_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered adapter.
        
        Args:
            adapter_id: Identifier of the adapter
            
        Returns:
            dict: Adapter information including name, type, and configuration schema
                  Returns None if adapter is not registered
        """
        if adapter_id not in self._adapters:
            return None

        try:
            # Create a temporary instance to get info (don't store it)
            adapter = self._adapters[adapter_id]()

            return {
                "id": adapter_id,
                "name": adapter.name,
                "type": adapter.adapter_type,
                "class": adapter.__class__.__name__,
                "configuration_schema": adapter.get_configuration_schema().get_schema(),
            }
        except Exception as e:
            self._logger.error(f"Error getting info for adapter '{adapter_id}': {e}")
            return None

    def set_adapter_config(self, adapter_id: str, config: Dict[str, Any]) -> None:
        """
        Set configuration for an adapter.
        
        This configuration will be used when initializing the adapter if not otherwise
        specified in get_adapter().
        
        Args:
            adapter_id: Identifier of the adapter
            config: Configuration dictionary
        """
        self._adapter_configs[adapter_id] = config
        self._logger.debug(f"Set configuration for adapter: {adapter_id}")

    def get_adapter_config(self, adapter_id: str) -> Dict[str, Any]:
        """
        Get the cached configuration for an adapter.
        
        Args:
            adapter_id: Identifier of the adapter
            
        Returns:
            dict: Configuration dictionary (empty dict if not set)
        """
        return self._adapter_configs.get(adapter_id, {})

    def clear_instances(self) -> None:
        """
        Clear all cached adapter instances.
        
        This will clean up resources for all initialized adapters.
        """
        for adapter_id, adapter in list(self._instances.items()):
            try:
                adapter.cleanup()
                self._logger.debug(f"Cleaned up adapter: {adapter_id}")
            except Exception as e:
                self._logger.error(f"Error cleaning up adapter '{adapter_id}': {e}")

        self._instances.clear()
        self._logger.info("Cleared all adapter instances")

    def is_adapter_initialized(self, adapter_id: str) -> bool:
        """
        Check if an adapter is currently initialized.
        
        Args:
            adapter_id: Identifier of the adapter
            
        Returns:
            bool: True if adapter is initialized, False otherwise
        """
        return adapter_id in self._instances and self._instances[adapter_id].is_initialized

    def list_all_adapters(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered adapters.
        
        Returns:
            List[Dict]: List of adapter information dictionaries
        """
        adapters_info = []
        for adapter_id in self.get_available_adapters():
            info = self.get_adapter_info(adapter_id)
            if info:
                adapters_info.append(info)
        return adapters_info

    def _initialize_default_adapters(self) -> None:
        """
        Initialize the registry with default built-in adapters.
        
        This is called automatically when the singleton is first created.
        """
        try:
            from abogen.tts_adapters.kokoro_adapter import KokoroAdapter

            self.register_adapter("kokoro", KokoroAdapter)
            self._logger.info("Registered default adapters")
        except ImportError as e:
            self._logger.error(f"Failed to import default adapters: {e}")

    def __del__(self):
        """Clean up resources when registry is destroyed"""
        self.clear_instances()


def get_tts_adapter(
    adapter_id: str,
    config: Optional[Dict[str, Any]] = None,
    force_reinit: bool = False,
) -> TTSAdapter:
    """
    Convenience function to get a TTS adapter from the global registry.
    
    Args:
        adapter_id: Identifier of the adapter
        config: Optional configuration for the adapter
        force_reinit: If True, reinitialize even if already initialized
        
    Returns:
        TTSAdapter: The adapter instance
        
    Raises:
        ValueError: If adapter is not registered
        TTSAdapterInitError: If adapter initialization fails
    """
    registry = TTSAdapterRegistry.get_instance()
    return registry.get_adapter(adapter_id, config=config, force_reinit=force_reinit)


def get_available_tts_adapters() -> List[str]:
    """
    Get list of available TTS adapters.
    
    Returns:
        List[str]: List of adapter IDs
    """
    registry = TTSAdapterRegistry.get_instance()
    return registry.get_available_adapters()


def list_tts_adapters() -> List[Dict[str, Any]]:
    """
    Get detailed information about all available TTS adapters.
    
    Returns:
        List[Dict]: List of adapter information
    """
    registry = TTSAdapterRegistry.get_instance()
    return registry.list_all_adapters()
