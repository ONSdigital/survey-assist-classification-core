"""Merged ClassificationLLM entry point for SIC and SOC domains."""

# pylint: disable=too-few-public-methods

from typing import Any, Literal

from survey_assist_classification_core.llm.sic_llm import (
    ClassificationLLM as SicClassificationLLM,
)
from survey_assist_classification_core.llm.soc_llm import (
    ClassificationLLM as SocClassificationLLM,
)


class ClassificationLLM:
    """Facade over domain-specific LLM orchestration for SIC and SOC.

    Selects the underlying implementation via ``classification_type`` while
    preserving the public method surface used by ``survey-assist-api`` today.
    """

    def __init__(
        self,
        classification_type: Literal["sic", "soc"] = "sic",
        **kwargs: Any,
    ) -> None:
        """Initialise the domain-specific LLM delegate."""
        if classification_type == "soc":
            self._delegate: SicClassificationLLM | SocClassificationLLM = (
                SocClassificationLLM(**kwargs)
            )
        else:
            self._delegate = SicClassificationLLM(**kwargs)
        self.classification_type = classification_type

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)
