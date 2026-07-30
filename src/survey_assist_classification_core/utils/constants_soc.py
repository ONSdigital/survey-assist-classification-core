"""SOC domain default configuration for Survey Assist classification-core."""

from survey_assist_classification_core.models.config_model import (
    SocFullConfig as FullConfig,
)


def get_default_config() -> FullConfig:
    """Return SOC defaults in the same structural shape as sic-classification-utils.

    Top-level keys are ``embedding``, ``llm``, and ``lookups`` (see ``FullConfig``).

    Returns:
        FullConfig: Embedding model path, generative LLM defaults, and SOC lookup paths.
    """
    return {
        "embedding": {
            "embedding_model_name": "all-MiniLM-L6-v2",
            "db_dir": "src/occupational_classification_utils/data/vector_store",
            "k_matches": 20,
        },
        "llm": {
            "llm_model_name": "gemini-2.5-flash",
            "model_location": "europe-west2",
            "code_digits": 4,
            "candidates_limit": 10,
        },
        "lookups": {
            "soc_index": (
                "occupational_classification_utils.data.soc_index",
                "soc2020volume2thecodingindexexcel16102024.xlsx",
            ),
            "soc_structure": (
                "occupational_classification_utils.data.soc_index",
                "soc2020volume1structureanddescriptionofunitgroupsexcel16102024.xlsx",
            ),
        },
    }
