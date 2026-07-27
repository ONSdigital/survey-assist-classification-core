"""LLM orchestration package for Survey Assist classification."""

# pylint: disable=duplicate-code

from survey_assist_classification_core.llm.prompt import (
    FIX_PARSING_PROMPT,
    SA_SIC_PROMPT_RAG,
    SIC_PROMPT_FINAL_ASSIGNMENT,
    SIC_PROMPT_OPENFOLLOWUP,
    SIC_PROMPT_UNAMBIGUOUS,
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
