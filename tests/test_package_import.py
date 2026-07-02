"""Tests for package and subpackage imports."""

from unittest.mock import MagicMock

import survey_assist_classification_core
from survey_assist_classification_core import config, llm, models
from survey_assist_classification_core.config import LlmDomainConfig, get_config
from survey_assist_classification_core.llm import ClassificationLLM


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


def test_classification_llm_facade_supports_sic_and_soc() -> None:
    """Merged ClassificationLLM exposes domain-specific methods via config."""
    mock_llm = MagicMock()
    sic = ClassificationLLM(classification_type="sic", llm=mock_llm)
    soc = ClassificationLLM(classification_type="soc", llm=mock_llm)
    assert hasattr(sic, "unambiguous_sic_code")
    assert hasattr(sic, "reranker_sic")
    assert hasattr(soc, "unambiguous_soc_code")
    assert hasattr(soc, "sa_rag_soc_code")
    assert not hasattr(sic, "unambiguous_soc_code")
    assert not hasattr(soc, "reranker_sic")


def test_get_config_returns_domain_lookups() -> None:
    """Domain config includes the expected lookup keys."""
    sic_config = get_config("sic")
    soc_config = get_config("soc")
    assert "sic_index" in sic_config["lookups"]
    assert "soc_index" in soc_config["lookups"]
