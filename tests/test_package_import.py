"""Tests for package and subpackage imports."""

import survey_assist_classification_core
import survey_assist_classification_core.config
import survey_assist_classification_core.llm
import survey_assist_classification_core.models
from survey_assist_classification_core.config import LlmDomainConfig


def test_package_imports() -> None:
    """Test that the root package can be imported."""
    assert survey_assist_classification_core is not None


def test_subpackages_import() -> None:
    """Test that LLM scaffold subpackages are importable."""
    assert survey_assist_classification_core.llm is not None
    assert survey_assist_classification_core.models is not None
    assert survey_assist_classification_core.config is not None


def test_llm_domain_config_stub() -> None:
    """Test the stub LlmDomainConfig model."""
    config = LlmDomainConfig(
        classification_type="sic",
        llm_model_name="gemini-2.5-flash",
    )
    assert config.classification_type == "sic"
    assert config.llm_model_name == "gemini-2.5-flash"
    assert config.model_location == "europe-west2"
    assert config.prompt_paths == {}
