"""Shared constants for Survey Assist classification-core."""

from survey_assist_classification_core.utils.constants_sic import (
    DEFAULT_TRUNCATE_LEN,
    MAX_ALT_CANDIDATES,
    truncate_identifier,
)
from survey_assist_classification_core.utils.constants_sic import (
    get_default_config as get_sic_default_config,
)
from survey_assist_classification_core.utils.constants_soc import (
    get_default_config as get_soc_default_config,
)

__all__ = [
    "DEFAULT_TRUNCATE_LEN",
    "MAX_ALT_CANDIDATES",
    "get_default_config",
    "truncate_identifier",
]


def get_default_config(classification_type: str = "sic"):
    """Return default configuration for the given classification domain."""
    if classification_type == "soc":
        return get_soc_default_config()
    return get_sic_default_config()
