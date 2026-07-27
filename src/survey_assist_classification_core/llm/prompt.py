"""Prompt templates for SIC and SOC classification LLM flows."""

from survey_assist_classification_core.llm.prompt_common import FIX_PARSING_PROMPT
from survey_assist_classification_core.llm.prompt_sic import (
    SA_SIC_PROMPT_RAG,
    SIC_PROMPT_FINAL_ASSIGNMENT,
    SIC_PROMPT_OPENFOLLOWUP,
    SIC_PROMPT_UNAMBIGUOUS,
)
from survey_assist_classification_core.llm.prompt_soc import (
    SOC_PROMPT_OPENFOLLOWUP,
    SOC_PROMPT_UNAMBIGUOUS,
)

__all__ = [
    "FIX_PARSING_PROMPT",
    "SA_SIC_PROMPT_RAG",
    "SIC_PROMPT_FINAL_ASSIGNMENT",
    "SIC_PROMPT_OPENFOLLOWUP",
    "SIC_PROMPT_UNAMBIGUOUS",
    "SOC_PROMPT_OPENFOLLOWUP",
    "SOC_PROMPT_UNAMBIGUOUS",
]
