"""Tests for package and subpackage imports."""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

import survey_assist_classification_core
from survey_assist_classification_core import config, llm, models
from survey_assist_classification_core.config import LlmDomainConfig, get_config
from survey_assist_classification_core.llm import (
    FIX_PARSING_PROMPT,
    SIC_PROMPT_UNAMBIGUOUS,
    SOC_PROMPT_UNAMBIGUOUS,
    ClassificationLLM,
)
from survey_assist_classification_core.models import (
    RagResponse,
    SicResponse,
    SocResponse,
    UnambiguousResponse,
)
from survey_assist_classification_core.utils.constants import get_default_config


def test_package_imports() -> None:
    """Test that the root package can be imported."""
    assert survey_assist_classification_core is not None


def test_subpackages_import() -> None:
    """Test that LLM scaffold subpackages are importable."""
    assert llm is not None
    assert models is not None
    assert config is not None


def test_llm_domain_config_stub() -> None:
    """Test the stub LlmDomainConfig model."""
    domain_config = LlmDomainConfig(
        classification_type="sic",
        llm_model_name="gemini-2.5-flash",
    )
    assert domain_config.classification_type == "sic"
    assert domain_config.llm_model_name == "gemini-2.5-flash"
    assert domain_config.model_location == "europe-west2"
    assert domain_config.prompt_paths == {}


def test_prompts_import_from_llm_package() -> None:
    """Merged prompts are importable without legacy utils."""
    assert FIX_PARSING_PROMPT is not None
    assert SIC_PROMPT_UNAMBIGUOUS is not None
    assert SOC_PROMPT_UNAMBIGUOUS is not None


def test_response_models_import_from_models_package() -> None:
    """Merged response models are importable without legacy utils."""
    assert SicResponse is not None
    assert SocResponse is not None
    assert RagResponse is not None
    assert UnambiguousResponse is not None


def test_get_default_config_returns_domain_lookups() -> None:
    """Domain config includes the expected lookup keys."""
    sic_config = get_default_config("sic")
    soc_config = get_default_config("soc")
    assert "sic_index" in sic_config["lookups"]
    assert "soc_index" in soc_config["lookups"]


def test_classification_llm_supports_sic_and_soc() -> None:
    """Merged ClassificationLLM exposes domain-specific methods via config."""
    mock_llm = MagicMock()
    sic = ClassificationLLM(classification_type="sic", llm=mock_llm)
    soc = ClassificationLLM(classification_type="soc", llm=mock_llm)
    assert hasattr(sic, "unambiguous_sic_code")
    assert hasattr(sic, "sa_rag_sic_code")
    assert hasattr(sic, "final_sic_code")
    assert hasattr(sic, "formulate_open_question")
    assert hasattr(soc, "unambiguous_soc_code")
    assert hasattr(soc, "top_one_soc_code")
    assert hasattr(soc, "formulate_open_question")
    assert not hasattr(sic, "unambiguous_soc_code")
    assert not hasattr(soc, "sa_rag_sic_code")


@pytest.mark.llm
async def test_classification_type_instantiates_domain_unambiguous(mocker) -> None:
    """classification_type selects the domain method used for unambiguous coding."""
    payload = {
        "codable": False,
        "class_code": None,
        "class_descriptive": None,
        "alt_candidates": [
            {
                "class_code": "1111",
                "class_descriptive": "description",
                "likelihood": 0.5,
            }
        ],
        "reasoning": "This is reasoning for the llm answer. Padded to 50 characters (Pydantic)",
    }
    mock_message = mocker.Mock(spec=AIMessage)
    mock_message.content = json.dumps(payload)
    mocker.patch(
        "survey_assist_classification_core.llm.llm.ChatVertexAI.ainvoke",
        return_value=mock_message,
    )

    sic = ClassificationLLM(classification_type="sic", model_name="gemini-2.5-flash")
    soc = ClassificationLLM(classification_type="soc", model_name="gemini-2.5-flash")

    sic_result = await sic.unambiguous_sic_code(
        industry_descr="school",
        semantic_search_results=[],
        job_title="teacher",
        job_description="educate kids",
    )
    soc_result = await soc.unambiguous_soc_code(
        industry_descr="school",
        semantic_search_results=[],
        job_title="teacher",
        job_description="educate kids",
        level_of_education="degree",
    )
    assert isinstance(sic_result[0], UnambiguousResponse)
    assert isinstance(soc_result[0], UnambiguousResponse)
    assert soc_result[1]["level_of_education"] == "degree"


def test_classification_llm_rejects_unknown_classification_type() -> None:
    """Invalid classification_type raises rather than defaulting silently."""
    mock_llm = MagicMock()
    try:
        ClassificationLLM(classification_type="nope", llm=mock_llm)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "classification_type must be 'sic' or 'soc'" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown classification_type")


def test_get_config_returns_domain_lookups() -> None:
    """Domain config includes the expected lookup keys."""
    sic_config = get_config("sic")
    soc_config = get_config("soc")
    assert "sic_index" in sic_config["lookups"]
    assert "soc_index" in soc_config["lookups"]
