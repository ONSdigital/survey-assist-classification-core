"""SIC domain default configuration for Survey Assist classification-core."""

from survey_assist_classification_core.models.config_model import (
    EmbeddingConfig,
    FullConfig,
    LLMConfig,
)


def get_default_config() -> FullConfig:
    """Returns the configuration dictionary for the LLM.

    Returns:
        FullConfig: A dictionary containing configuration details for the embedding model
        and lookup file paths.
    """
    return {
        "embedding": EmbeddingConfig(
            embedding_model_name="all-MiniLM-L6-v2",  # text-embedding-004
            db_dir="src/industrial_classification_utils/data/vector_store",
            index_source_file="src/industrial_classification_utils/data/sic_index/"
            + "uksic2007indexeswithaddendumdecember2022.csv",
            k_matches=20,
        ),
        "llm": LLMConfig(
            llm_model_name="gemini-2.5-flash",
            model_location="europe-west2",
            code_digits=5,
            candidates_limit=10,
        ),
        "lookups": {
            "sic_index": (
                "industrial_classification_utils.data.sic_index",
                # "extended_SIC_index.xlsx",
                "uksic2007indexeswithaddendumdecember2022.xlsx",
            ),
            "sic_structure": (
                "industrial_classification_utils.data.sic_index",
                "publisheduksicsummaryofstructureworksheet.xlsx",
            ),
            "sic_condensed": (
                "industrial_classification_utils.data.example",
                "sic_2d_condensed.txt",
            ),
        },
    }
