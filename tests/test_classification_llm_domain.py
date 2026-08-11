"""Domain-boundary tests for the merged ClassificationLLM.

Covers SIC/SOC instantiation, domain-specific prompt attributes, and
cross-domain AttributeError rejection.
"""

from unittest.mock import MagicMock

import pytest

from survey_assist_classification_core.llm import (
    SA_SIC_PROMPT_RAG,
    SIC_PROMPT_FINAL_ASSIGNMENT,
    SIC_PROMPT_OPENFOLLOWUP,
    SIC_PROMPT_UNAMBIGUOUS,
    SOC_PROMPT_OPENFOLLOWUP,
    SOC_PROMPT_UNAMBIGUOUS,
    ClassificationLLM,
)

_SIC_PUBLIC_METHODS = (
    "sa_rag_sic_code",
    "unambiguous_sic_code",
    "final_sic_code",
)
_SOC_PUBLIC_METHODS = ("unambiguous_soc_code",)
_SIC_PROMPTS = (
    "sa_sic_prompt_rag",
    "sic_prompt_unambiguous",
    "sic_prompt_openfollowup",
    "sic_prompt_final",
)
_SOC_PROMPTS = (
    "soc_prompt_unambiguous",
    "soc_prompt_openfollowup",
)


def test_sic_classification_llm_instantiates_with_prompts() -> None:
    """SIC ClassificationLLM wires SIC prompts and keeps the injected LLM."""
    mock_llm = MagicMock()
    sic = ClassificationLLM(classification_type="sic", llm=mock_llm)

    assert sic.classification_type == "sic"
    assert sic.llm is mock_llm
    assert sic.sa_sic_prompt_rag is SA_SIC_PROMPT_RAG
    assert sic.sic_prompt_unambiguous is SIC_PROMPT_UNAMBIGUOUS
    assert sic.sic_prompt_openfollowup is SIC_PROMPT_OPENFOLLOWUP
    assert sic.sic_prompt_final is SIC_PROMPT_FINAL_ASSIGNMENT
    assert callable(sic.formulate_open_question)
    for method_name in _SIC_PUBLIC_METHODS:
        assert callable(getattr(sic, method_name))


def test_soc_classification_llm_instantiates_with_prompts() -> None:
    """SOC ClassificationLLM wires SOC prompts and keeps the injected LLM."""
    mock_llm = MagicMock()
    soc = ClassificationLLM(classification_type="soc", llm=mock_llm)

    assert soc.classification_type == "soc"
    assert soc.llm is mock_llm
    assert soc.soc_prompt_unambiguous is SOC_PROMPT_UNAMBIGUOUS
    assert soc.soc_prompt_openfollowup is SOC_PROMPT_OPENFOLLOWUP
    assert callable(soc.formulate_open_question)
    for method_name in _SOC_PUBLIC_METHODS:
        assert callable(getattr(soc, method_name))


@pytest.mark.parametrize(
    "attr_name", [*_SIC_PUBLIC_METHODS, *_SIC_PROMPTS, "sic_meta", "sic"]
)
def test_soc_classification_llm_rejects_sic_attributes(attr_name: str) -> None:
    """SOC ClassificationLLM raises AttributeError for SIC-only attributes."""
    soc = ClassificationLLM(classification_type="soc", llm=MagicMock())

    with pytest.raises(AttributeError, match=attr_name):
        getattr(soc, attr_name)


@pytest.mark.parametrize(
    "attr_name", [*_SOC_PUBLIC_METHODS, *_SOC_PROMPTS, "soc_meta", "soc"]
)
def test_sic_classification_llm_rejects_soc_attributes(attr_name: str) -> None:
    """SIC ClassificationLLM raises AttributeError for SOC-only attributes."""
    sic = ClassificationLLM(classification_type="sic", llm=MagicMock())

    with pytest.raises(AttributeError, match=attr_name):
        getattr(sic, attr_name)
