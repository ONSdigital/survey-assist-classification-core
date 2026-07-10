"""LLM orchestration for Survey Assist classification."""

# pylint: disable=duplicate-code

from survey_assist_classification_core.llm.llm import ClassificationLLM
from survey_assist_classification_core.llm.prompt import (
    FIX_PARSING_PROMPT,
    GENERAL_PROMPT_RAG,
    SA_SIC_PROMPT_RAG,
    SA_SOC_PROMPT_RAG,
    SIC_PROMPT_CLOSEDFOLLOWUP,
    SIC_PROMPT_FINAL_ASSIGNMENT,
    SIC_PROMPT_OPENFOLLOWUP,
    SIC_PROMPT_PYDANTIC,
    SIC_PROMPT_RAG,
    SIC_PROMPT_RERANKER,
    SIC_PROMPT_UNAMBIGUOUS,
    SOC_PROMPT_OPENFOLLOWUP,
    SOC_PROMPT_PYDANTIC,
    SOC_PROMPT_UNAMBIGUOUS,
)
from survey_assist_classification_core.llm.sic_llm import (
    ClassificationLLM as SicClassificationLLM,
)
from survey_assist_classification_core.llm.soc_llm import (
    ClassificationLLM as SocClassificationLLM,
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
    "ClassificationLLM",
    "SicClassificationLLM",
    "SocClassificationLLM",
]
