"""Prompt templates for SIC and SOC classification LLM flows."""

from survey_assist_classification_core.llm.prompt_common import FIX_PARSING_PROMPT
from survey_assist_classification_core.llm.prompt_sic import (
    GENERAL_PROMPT_RAG,
    SA_SIC_PROMPT_RAG,
    SIC_PROMPT_CLOSEDFOLLOWUP,
    SIC_PROMPT_FINAL_ASSIGNMENT,
    SIC_PROMPT_OPENFOLLOWUP,
    SIC_PROMPT_PYDANTIC,
    SIC_PROMPT_RAG,
    SIC_PROMPT_RERANKER,
    SIC_PROMPT_UNAMBIGUOUS,
)
from survey_assist_classification_core.llm.prompt_soc import (
    SA_SOC_PROMPT_RAG,
    SOC_PROMPT_OPENFOLLOWUP,
    SOC_PROMPT_PYDANTIC,
    SOC_PROMPT_UNAMBIGUOUS,
)

__all__ = [
    "FIX_PARSING_PROMPT",
    "GENERAL_PROMPT_RAG",
    "SA_SIC_PROMPT_RAG",
    "SA_SOC_PROMPT_RAG",
    "SIC_PROMPT_CLOSEDFOLLOWUP",
    "SIC_PROMPT_FINAL_ASSIGNMENT",
    "SIC_PROMPT_OPENFOLLOWUP",
    "SIC_PROMPT_PYDANTIC",
    "SIC_PROMPT_RAG",
    "SIC_PROMPT_RERANKER",
    "SIC_PROMPT_UNAMBIGUOUS",
    "SOC_PROMPT_OPENFOLLOWUP",
    "SOC_PROMPT_PYDANTIC",
    "SOC_PROMPT_UNAMBIGUOUS",
]
