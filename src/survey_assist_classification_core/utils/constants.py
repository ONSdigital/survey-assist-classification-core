"""Shared constants for Survey Assist classification-core."""

from survey_assist_classification_core.utils.constants_sic import (
    get_default_config as get_sic_default_config,
)
from survey_assist_classification_core.utils.constants_soc import (
    get_default_config as get_soc_default_config,
)

MAX_ALT_CANDIDATES = 10
DEFAULT_TRUNCATE_LEN = 8

__all__ = [
    "DEFAULT_TRUNCATE_LEN",
    "MAX_ALT_CANDIDATES",
    "get_default_config",
    "truncate_identifier",
]


def truncate_identifier(value: str | None, max_len: int = DEFAULT_TRUNCATE_LEN) -> str:
    """Return a truncated string safely, handling None and short values.

    Used for logging to preserve privacy while providing enough context.

    Args:
        value (str | None): The string to truncate.
        max_len (int): Maximum length before truncation. Defaults to 8.

    Returns:
        str: Empty string if value is None/empty, otherwise truncated string
            with "..." suffix if longer than max_len.
    """
    if not value:
        return ""
    return value if len(value) <= max_len else value[:max_len] + "..."


def get_default_config(classification_type: str = "sic"):
    """Return default configuration for the given classification domain."""
    if classification_type == "soc":
        return get_soc_default_config()
    return get_sic_default_config()
