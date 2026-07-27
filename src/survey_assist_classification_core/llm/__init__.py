"""LLM orchestration for Survey Assist classification."""

# pylint: disable=duplicate-code

from survey_assist_classification_core.llm.llm import ClassificationLLM
from survey_assist_classification_core.llm.prompt import (
    FIX_PARSING_PROMPT,
    SA_SIC_PROMPT_RAG,
    SIC_PROMPT_FINAL_ASSIGNMENT,
    SIC_PROMPT_OPENFOLLOWUP,
    SIC_PROMPT_UNAMBIGUOUS,
    SOC_PROMPT_OPENFOLLOWUP,
    SOC_PROMPT_UNAMBIGUOUS,
)


class SicClassificationLLM(ClassificationLLM):
    """SIC-domain ClassificationLLM (``classification_type="sic"``)."""

    def __init__(self, **kwargs) -> None:
        """Initialise a SIC ClassificationLLM."""
        super().__init__(classification_type="sic", **kwargs)


class SocClassificationLLM(ClassificationLLM):
    """SOC-domain ClassificationLLM (``classification_type="soc"``)."""

    def __init__(self, **kwargs) -> None:
        """Initialise a SOC ClassificationLLM."""
        super().__init__(classification_type="soc", **kwargs)


__all__ = [
    "FIX_PARSING_PROMPT",
    "SA_SIC_PROMPT_RAG",
    "SIC_PROMPT_FINAL_ASSIGNMENT",
    "SIC_PROMPT_OPENFOLLOWUP",
    "SIC_PROMPT_UNAMBIGUOUS",
    "SOC_PROMPT_OPENFOLLOWUP",
    "SOC_PROMPT_UNAMBIGUOUS",
    "ClassificationLLM",
    "SicClassificationLLM",
    "SocClassificationLLM",
]
