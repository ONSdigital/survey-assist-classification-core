"""Domain configuration models for Survey Assist classification."""

from pydantic import BaseModel, Field


class LlmDomainConfig(BaseModel):
    """Stub configuration for domain-specific LLM settings.

    Attributes:
        classification_type: Classification domain identifier (for example ``sic`` or
            ``soc``).
        llm_model_name: Name of the language model to use.
        model_location: Cloud region for hosted models.
        prompt_paths: Mapping of prompt identifiers to file paths.
    """

    classification_type: str
    llm_model_name: str
    model_location: str = "europe-west2"
    prompt_paths: dict[str, str] = Field(default_factory=dict)
