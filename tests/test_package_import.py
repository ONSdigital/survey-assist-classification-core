"""Tests for package and subpackage imports."""

import survey_assist_classification_core
from survey_assist_classification_core import config, llm, models
from survey_assist_classification_core.config import LlmDomainConfig
from survey_assist_classification_core.llm import (
    FIX_PARSING_PROMPT,
    SIC_PROMPT_UNAMBIGUOUS,
    SOC_PROMPT_UNAMBIGUOUS,
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
