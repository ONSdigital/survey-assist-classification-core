"""Domain configuration models for Survey Assist classification."""

from pydantic import BaseModel


class LlmDomainConfig(BaseModel):
    """Stub configuration for domain-specific LLM settings.

    Attributes:
        classification_type: Classification domain identifier (for example ``sic`` or
            ``soc``).
        llm_model_name: Name of the language model to use.
        model_location: Cloud region for hosted models.
        code_digits: Number of digits in the classification code.
        candidates_limit: Maximum number of candidate codes to return.
    """

    classification_type: str
    llm_model_name: str
    model_location: str = "europe-west2"
    code_digits: int
    candidates_limit: int
