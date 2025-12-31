import re
from abogen.voice_manager import VoiceManager
from abogen.tts_adapters.registry import TTSAdapterRegistry
from abogen.tts_settings import get_tts_config_manager


# Calls parsing and loads the voice to gpu or cpu
def get_new_voice(tts_pipeline_or_adapter, formula, use_gpu):
    """
    Load a voice based on a formula (which may include weighted voice mixing).
    
    Args:
        tts_pipeline_or_adapter: Either a wrapper function or a TTS adapter instance
        formula: Voice formula string (e.g., "af_alloy" or "af_alloy*0.5+am_blake*0.5")
        use_gpu: Whether to use GPU
        
    Returns:
        Voice tensor ready for use by the TTS pipeline
    """
    try:
        weighted_voice = parse_voice_formula(tts_pipeline_or_adapter, formula)
        # device = "cuda" if use_gpu else "cpu"
        # Setting the device "cuda" gives "Error occurred: split_with_sizes(): argument 'split_sizes' (position 2)"
        # error when the device is gpu. So disabling this for now.
        device = "cpu"
        return weighted_voice.to(device)
    except Exception as e:
        raise ValueError(f"Failed to create voice: {str(e)}")


# Parse the formula and get the combined voice tensor
def parse_voice_formula(tts_pipeline_or_adapter, formula):
    """
    Parse a voice formula and return the combined voice tensor.
    
    Supports:
    - Simple voice IDs: "af_alloy"
    - Weighted formulas: "af_alloy*0.5+am_blake*0.5"
    
    Args:
        tts_pipeline_or_adapter: Either a Kokoro KPipeline, an adapter instance, or a wrapper function
        formula: Voice formula string
        
    Returns:
        Combined voice tensor (numpy array)
    """
    if not formula.strip():
        raise ValueError("Empty voice formula")

    # Initialize the weighted sum
    weighted_sum = None

    total_weight = calculate_sum_from_formula(formula)

    # Split the formula into terms
    voices = formula.split("+")

    # Get the actual adapter/pipeline for loading voices
    # If it's a wrapper function, get the adapter from the registry
    pipeline_instance = tts_pipeline_or_adapter
    if hasattr(tts_pipeline_or_adapter, '__self__'):
        # It's a bound method - get the adapter from the class
        if hasattr(tts_pipeline_or_adapter.__self__, 'adapter'):
            pipeline_instance = tts_pipeline_or_adapter.__self__.adapter
    elif callable(tts_pipeline_or_adapter) and not hasattr(tts_pipeline_or_adapter, 'load_single_voice'):
        # It's a wrapper function - get the active adapter
        try:
            registry = TTSAdapterRegistry.get_instance()
            config_manager = get_tts_config_manager()
            tts_config = config_manager.get_configuration()
            pipeline_instance = registry.get_adapter(tts_config.active_adapter)
        except Exception:
            # Fallback: try to get from conversion thread context
            pass

    for term in voices:
        # Parse each term (format: "voice_name*0.333")
        voice_name, weight = term.strip().split("*")
        weight = float(weight.strip())
        # normalize the weight
        weight /= total_weight if total_weight > 0 else 1.0
        voice_name = voice_name.strip()

        # Get the voice tensor
        if not VoiceManager.is_valid_voice(voice_name):
            raise ValueError(f"Unknown voice: {voice_name}")

        voice_tensor = pipeline_instance.load_single_voice(voice_name)

        # Add to weighted sum
        if weighted_sum is None:
            weighted_sum = weight * voice_tensor
        else:
            weighted_sum += weight * voice_tensor

    return weighted_sum


def calculate_sum_from_formula(formula):
    weights = re.findall(r"\* *([\d.]+)", formula)
    total_sum = sum(float(weight) for weight in weights)
    return total_sum
