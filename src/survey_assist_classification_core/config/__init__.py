"""Domain configuration for Survey Assist classification-core."""

from typing import Literal

from pydantic import BaseModel, Field


class LlmDomainConfig(BaseModel):
    """Configuration for domain-specific LLM settings.

    Attributes:
        classification_type: Classification domain identifier (``sic`` or ``soc``).
        llm_model_name: Name of the language model to use.
        model_location: Cloud region for hosted models.
        prompt_paths: Mapping of prompt identifiers to file paths.
    """

    classification_type: Literal["sic", "soc"]
    llm_model_name: str
    model_location: str = "europe-west2"
    prompt_paths: dict[str, str] = Field(default_factory=dict)


def get_config(classification_type: Literal["sic", "soc"] = "sic"):
    """Return full default configuration for a classification domain."""
    # Lazy import avoids a circular import with models.__init__ / constants_sic.
    from survey_assist_classification_core.utils.constants import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        get_default_config,
    )

    return get_default_config(classification_type)
