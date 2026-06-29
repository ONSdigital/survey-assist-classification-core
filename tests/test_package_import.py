"""Tests for package and subpackage imports."""

import survey_assist_classification_core
from survey_assist_classification_core import config, llm, models
from survey_assist_classification_core.config import LlmDomainConfig


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
