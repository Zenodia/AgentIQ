from pydantic import AliasChoices
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PositiveInt

from aiq.builder.builder import Builder
from aiq.builder.llm import LLMProviderInfo
from aiq.cli.register_workflow import register_llm_provider
from aiq.data_models.llm import LLMBaseConfig

class HuggingFaceModelConfig(LLMBaseConfig, name="huggingface"):
    """A HuggingFace LLM provider configuration."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str = Field(
        description="The name/path of the HuggingFace model"
    )
    model_type: str = Field(
        description="Model type, currently support only AutoModelForCausalLM and AutoModelForSequenceClassification,"
    )
    device: str = Field(
        default="cuda:0",
        description="Device to run the model on (cuda:0 or cpu)"
    )
    temperature: float = Field(
        default=0.6,
        description="Sampling temperature in [0, 1]"
    )
    top_p: float = Field(
        default=1.0,
        description="Top-p for nucleus sampling"
    )
    max_new_tokens: int = Field(
        default=100,
        description="Maximum number of tokens to generate"
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for generation"
    )

    
@register_llm_provider(config_type=HuggingFaceModelConfig)
async def register_huggingface_provider(config: HuggingFaceModelConfig, builder: Builder):
    """Register HuggingFace LLM provider."""
    try:
        # Validate dependencies
        import torch
        from transformers import AutoTokenizer
        
        # Pre-flight check
        AutoTokenizer.from_pretrained(config.model_name)
        
        yield LLMProviderInfo(
            config=config,
            description="HuggingFace LLM provider for local model inference"
        )
    finally:
        # Cleanup if needed
        torch.cuda.empty_cache()


    