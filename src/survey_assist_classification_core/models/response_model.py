"""This module defines response models for industrial classification utilities.

The models are implemented using Pydantic's `BaseModel` and are used to represent
various response structures for SIC (Standard Industrial Classification) code
assignment and classification tasks. These models include validation logic and
field-level constraints to ensure data integrity.

Classes:
    SicCandidate: Represents a candidate SIC code with associated information.
    SicResponse: Represents a response model for SIC code assignment.
    RagCandidate: Represents a candidate classification code with associated information.
    UnambiguousResponse: Represents a response model for unambiguous
                         classification code assignment.
    FinalSICAssignment: Response model for final assignment of a SIC code.
    OpenFollowUp: Represents a response model for open ended follow-up question.
<<<<<<< HEAD
    TopOneResponse: Top-ranked SOC code selected from a supplied shortlist.
=======
>>>>>>> origin/main

Constants:
    MAX_ALT_CANDIDATES: Maximum number of alternative candidates allowed in certain models.
"""

from pydantic import BaseModel, Field, field_validator, model_validator

from survey_assist_classification_core.utils.constants import MAX_ALT_CANDIDATES


class SicCandidate(BaseModel):
    """Represents a candidate SIC code with associated information.

    Attributes:
        sic_code (str): Plausible SIC code based on the company activity description.
        sic_descriptive (str): Descriptive label of the SIC category associated with
            sic_code.
        likelihood (float): Likelihood of this sic_code with a value between 0 and 1.

    """

    sic_code: str = Field(
        description="Plausible SIC code based on the company activity description."
    )
    sic_descriptive: str = Field(
        description="Descriptive label of the SIC category associated with sic_code."
    )
    likelihood: float = Field(
        description="Likelihood of this sic_code with value between 0 and 1."
    )


class SicResponse(BaseModel):
    """Represents a response model for SIC code assignment.

    Attributes:
        codable (bool): True if enough information is provided to decide SIC code,
            False otherwise.
        followup (Optional[str]): Question to ask user in order to collect additional
            information to enable reliable SIC assignment. Empty if codable=True.
        sic_code (Optional[str]): Full SIC code (to the required number of digits)
            assigned based on the provided company activity description.
            Empty if codable=False.
        sic_descriptive (Optional[str]): Descriptive label of the SIC category
            associated with sic_code if provided. Empty if codable=False.
        sic_candidates (List[SicCandidate]): Short list of less than ten possible or
            alternative sic codes that may be applicable with their descriptive label
            and estimated likelihood.
        sic_code_2digits (Optional[str]): First two digits of the hierarchical SIC
            code assigned. This field should be non empty if the larger (two-digit)
            group of SIC codes can be determined even in cases where additional
            information is needed to code to four digits (for example when all
            SIC candidates share the same first two digits).
        reasoning (str): Specifies the information used to assign the SIC code or any
            additional information required to assign a SIC code.
    """

    codable: bool = Field(
        description="""True if enough information is provided to decide
        SIC code, False otherwise.""",
        default=False,
    )
    followup: str | None = Field(
        description="""Question to ask user in order to collect additional information
        to enable reliable SIC assignment. Empty if codable=True.""",
        default=None,
    )
    sic_code: str | None = Field(
        description="""Full SIC code (to the required number of digits) assigned based
        on provided the company activity description.  Empty if codable=False.""",
        default=None,
    )
    sic_descriptive: str | None = Field(
        description="""Descriptive label of the SIC category associated with sic_code
        if provided. Empty if codable=False.""",
        default=None,
    )
    sic_candidates: list[SicCandidate] = Field(
        description="""Short list of less than ten possible or alternative SIC codes
        that may be applicable with their descriptive label and estimated likelihood.""",
        default=[],
    )

    reasoning: str = Field(
        description="""Step by step reasoning behind classification selected. Specifies
            the information used to assign the SIC code or any additional information
            required to assign a SIC code.""",
        default="No reasoning provided.",
    )

    @classmethod
    def sic_code_validator(cls, v):
        """Validates that a valid SIC code is provided if the response is codable.

        Args:
            v (str): The SIC code to validate.

        Returns:
            str: The validated SIC code.

        Raises:
            ValueError: If the SIC code is empty when codable is True.
        """
        if v == "":
            raise ValueError("If codable, then valid sic_code needs to be provided")
        return v

    @model_validator(mode="before")
    @classmethod
    def check_valid_fields(cls, values):
        """Validates the fields of the model before instantiation.

        Ensures that:
        - If `codable` is True, a valid `sic_code` is provided.
        - If `codable` is False, a follow-up question is provided.

        Args:
            values (dict): The dictionary of field values.

        Returns:
            dict: The validated field values.

        Raises:
            ValueError: If validation conditions are not met.
        """
        if values.get("codable"):
            cls.sic_code_validator(values.get("sic_code"))
        elif not values.get("followup"):  # This checks for None or empty string
            raise ValueError("If uncodable, a follow-up question needs to be provided.")
        return values


class RagCandidate(BaseModel):
    """Represents a candidate classification code with associated information.

    Attributes:
        class_code (str): Plausible classification code based on the respondent's data.
        class_descriptive (str): Descriptive label of the classification category
            associated with class_code.
        likelihood (float): Likelihood of this class_code with a value between 0 and 1.

    """

    class_code: str = Field(
        description="Plausible classification code based on the respondent's data."
    )
    class_descriptive: str = Field(
        description="""Descriptive label of the classification category
        associated with class_code."""
    )
    likelihood: float = Field(
        description="Likelihood of this class_code with value between 0 and 1."
    )


class UnambiguousResponse(BaseModel):
    """Represents a response model for classification code assignment.

    Attributes:
        codable (bool): True only if enough information is provided to assign
            an unambiguous single classification code, False otherwise.
        class_code (Optional[str]): Full classification code (to the required number of digits)
            assigned based on provided respondent's data. Must be present if codable=True,
            must be None if codable=False.
        class_descriptive (Optional[str]): Descriptive label of the classification category.
            Must be present if codable=True, must be None if codable=False.
        alt_candidates (list[RagCandidate]): Short list of possible classification codes with their
            descriptive labels and estimated likelihoods.
        reasoning (str): Step by step reasoning behind the classification selected.
    """

    codable: bool = Field(
        description="True only if enough information is provided to decide an unambiguous "
        "classification code, False otherwise."
    )

    class_code: str | None = Field(
        default=None,
        description="Full classification code (to the required number of digits) "
        "assigned based on provided respondent's data. Must be present if codable=True, "
        "must be None if codable=False.",
    )

    class_descriptive: str | None = Field(
        default=None,
        description="Descriptive label of the classification category. "
        "Must be present if codable=True, must be None if codable=False.",
    )

    alt_candidates: list[RagCandidate] = Field(
        default_factory=list,
        description="Short list of possible classification codes with their "
        "descriptive labels and estimated likelihoods.",
        min_length=1,  # Ensure there's always at least one candidate
        max_length=10,  # Limit to less than 10 candidates
    )

    reasoning: str = Field(
        description="Step by step reasoning behind the classification selected.",
        min_length=50,  # Ensure detailed reasoning is provided
    )

    @field_validator("alt_candidates")
    @classmethod
    def validate_alt_candidates(cls, v):
        """Validates the number of alternative candidates.

        Ensures that the number of candidates is between 1 and the maximum allowed.

        Args:
            v (list): The list of alternative candidates.

        Returns:
            list: The validated list of candidates.

        Raises:
            ValueError: If the number of candidates is not within the allowed range.
        """
        if not 1 <= len(v) <= MAX_ALT_CANDIDATES:
            raise ValueError("alt_candidates must contain between 1 and 10 items.")
        return v


class FinalSICAssignment(BaseModel):
    """Response model for final assignment of a SIC code.

    Attributes:
        codable (bool): True if enough information is provided to assign
            an unambiguous single 5-digit classification code, False otherwise.
        unambiguous_code (Optional[str]): Full 5-digit classification code
            assigned based on provided respondent's data. Must be present if codable=True,
            must be None if codable=False.
        unambiguous_code_descriptive (Optional[str]): Descriptive label of the classification
            category. Must be present if codable=True, must be None if codable=False.
        higher_level_code (Optional[str]): Classification code with X notation to pad to 5 digits.
            Must be present if codable=False, must be None if codable=True.
        reasoning (str): Step by step reasoning behind the classification selected.
    """

    codable: bool = Field(
        description="True only if enough information is provided to decide an unambiguous "
        "classification code, False otherwise."
    )
    unambiguous_code: str | None = Field(
        description="Full 5-digit classification code "
        "assigned based on provided respondent's data. Must be present if codable=True, "
        "must be None if codable=False."
    )
    unambiguous_code_descriptive: str | None = Field(
        description="Descriptive label of the classification category. "
        "Must be present if codable=True, must be None if codable=False."
    )
    higher_level_code: str | None = Field(
        description="Classification code with X notation to pad to 5 digits. "
        "Must be present if codable=False, must be None if codable=True."
    )
    reasoning: str = Field(
        description="Step by step reasoning behind the classification selected.",
        min_length=50,
    )


class OpenFollowUp(BaseModel):
    """Represents a response model for open ended follow-up question.

    Attributes:
        followup (str): Question to ask user in order to collect
            additional information to enable reliable classification assignment.
        reasoning (str): Reasoning explaining how follow-up question will help
            assign classification code.
    """

    followup: str | None = Field(
        description="""Question to ask user in order to collect additional information
        to enable reliable classification assignment.""",
        default="",
    )
    reasoning: str = Field(
        description="""Reasoning explaining how follow-up question will help
            assign classification code.""",
        default="",
    )
<<<<<<< HEAD


class TopOneResponse(BaseModel):
    """Top-ranked SOC code selected from a supplied shortlist."""

    soc_code: str = Field(
        description="Selected four-digit SOC code from the provided shortlist.",
        min_length=1,
    )
    soc_title: str = Field(
        description="Title label associated with the selected SOC code.",
        min_length=1,
    )
    likelihood_score: float = Field(
        description=(
            "Likelihood of the selected SOC code relative to the other shortlisted "
            "candidates, between 0 and 1."
        ),
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description=(
            "Reasoning explaining why the selected SOC code is the strongest "
            "match from the shortlist and why the likelihood is as reported."
        ),
        min_length=1,
    )
=======
>>>>>>> origin/main
