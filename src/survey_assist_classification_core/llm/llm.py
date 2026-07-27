# pylint: disable=logging-not-lazy,logging-fstring-interpolation,too-many-lines,duplicate-code
"""Merged ClassificationLLM for SIC and SOC classification domains.

Single class selected by ``classification_type``. Domain-specific public method
names used by Survey Assist today are preserved. Shared orchestration helpers
remove the previous parallel ``sic_llm`` / ``soc_llm`` duplication.
"""

from __future__ import annotations

import time
from collections import defaultdict
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from industrial_classification.data_access.sic_data_access import load_sic_hierarchy
from industrial_classification.hierarchy.sic_hierarchy import SIC
from industrial_classification.meta import sic_meta
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from occupational_classification.data_access.soc_data_access import (
    get_soc_meta,
    load_soc_hierarchy,
)
from occupational_classification.hierarchy.soc_hierarchy import SOC
from pydantic import SecretStr
from survey_assist_utils.logging import get_logger

from survey_assist_classification_core.config import get_config
from survey_assist_classification_core.llm.prompt import (
    FIX_PARSING_PROMPT,
    GENERAL_PROMPT_RAG,
    SA_SIC_PROMPT_RAG,
    SA_SOC_PROMPT_RAG,
    SIC_PROMPT_CLOSEDFOLLOWUP,
    SIC_PROMPT_FINAL_ASSIGNMENT,
    SIC_PROMPT_OPENFOLLOWUP,
    SIC_PROMPT_PYDANTIC,
    SIC_PROMPT_RAG,
    SIC_PROMPT_RERANKER,
    SIC_PROMPT_UNAMBIGUOUS,
    SOC_PROMPT_OPENFOLLOWUP,
    SOC_PROMPT_PYDANTIC,
    SOC_PROMPT_UNAMBIGUOUS,
)
from survey_assist_classification_core.models.response_model import (
    ClosedFollowUp,
    FinalSICAssignment,
    OpenFollowUp,
    RagCandidate,
    RerankingResponse,
    SicCandidate,
    SicResponse,
    SocResponse,
    UnambiguousResponse,
)
from survey_assist_classification_core.utils.constants import truncate_identifier

logger = get_logger(__name__)

_SIC_ONLY_ATTRS = frozenset(
    {
        "get_sic_code",
        "sa_rag_sic_code",
        "unambiguous_sic_code",
        "reranker_sic",
        "final_sic_code",
        "formulate_closed_question",
        "_prompt_candidate_list_filtered",
        "_prompt_candidate_sic",
        "sic_prompt",
        "sic_meta",
        "sic_prompt_rag",
        "sa_sic_prompt_rag",
        "general_prompt_rag",
        "sic_prompt_unambiguous",
        "sic_prompt_reranker",
        "sic_prompt_openfollowup",
        "sic_prompt_closedfollowup",
        "sic_prompt_final",
        "sic",
    }
)
_SOC_ONLY_ATTRS = frozenset(
    {
        "get_soc_code",
        "sa_rag_soc_code",
        "unambiguous_soc_code",
        "_prompt_candidate_soc",
        "soc_prompt",
        "soc_meta",
        "sa_soc_prompt_rag",
        "soc_prompt_unambiguous",
        "soc_prompt_openfollowup",
        "soc",
    }
)


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
class ClassificationLLM:
    """LLM orchestration for SIC and SOC classification via domain config."""

    def __init__(  # noqa: PLR0913
        self,
        classification_type: Literal["sic", "soc"] = "sic",
        model_name: str | None = None,
        model_location: str | None = None,
        llm: ChatVertexAI | ChatOpenAI | None = None,
        max_tokens: int = 1600,
        temperature: float = 0.0,
        verbose: bool = True,
        openai_api_key: SecretStr | None = None,
    ):
        """Initialise the ClassificationLLM for a classification domain."""
        if classification_type not in {"sic", "soc"}:
            raise ValueError(
                f"classification_type must be 'sic' or 'soc', got {classification_type!r}"
            )

        self.classification_type = classification_type
        self.config = get_config(classification_type)
        if model_name is None:
            model_name = self.config["llm"]["llm_model_name"]
        if model_location is None:
            model_location = self.config["llm"]["model_location"]

        logger.info(
            f"Init LLM {llm} model: {model_name} max_tokens: {max_tokens} temp: {temperature}"
        )
        self.llm = self._create_chat_model(
            model_name=model_name,
            model_location=model_location,
            llm=llm,
            max_tokens=max_tokens,
            temperature=temperature,
            openai_api_key=openai_api_key,
        )
        self.verbose = verbose
        self.sic: SIC | None = None
        self.soc: SOC | None = None

        if classification_type == "sic":
            self.sic_meta = sic_meta
            self.sic_prompt = SIC_PROMPT_PYDANTIC
            self.sic_prompt_rag = SIC_PROMPT_RAG
            self.sa_sic_prompt_rag = SA_SIC_PROMPT_RAG
            self.general_prompt_rag = GENERAL_PROMPT_RAG
            self.sic_prompt_unambiguous = SIC_PROMPT_UNAMBIGUOUS
            self.sic_prompt_reranker = SIC_PROMPT_RERANKER
            self.sic_prompt_openfollowup = SIC_PROMPT_OPENFOLLOWUP
            self.sic_prompt_closedfollowup = SIC_PROMPT_CLOSEDFOLLOWUP
            self.sic_prompt_final = SIC_PROMPT_FINAL_ASSIGNMENT
        else:
            self.soc_meta = get_soc_meta(self.config["lookups"]["soc_structure"])
            self.soc_prompt = SOC_PROMPT_PYDANTIC
            self.sa_soc_prompt_rag = SA_SOC_PROMPT_RAG
            self.soc_prompt_unambiguous = SOC_PROMPT_UNAMBIGUOUS
            self.soc_prompt_openfollowup = SOC_PROMPT_OPENFOLLOWUP

    def __getattribute__(self, name: str) -> Any:
        """Hide domain-only attributes for the inactive classification type."""
        if name.startswith("__") and name.endswith("__"):
            return object.__getattribute__(self, name)
        try:
            ctype = object.__getattribute__(self, "classification_type")
        except AttributeError:
            return object.__getattribute__(self, name)
        if name in _SIC_ONLY_ATTRS and ctype != "sic":
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        if name in _SOC_ONLY_ATTRS and ctype != "soc":
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return object.__getattribute__(self, name)

    def _require_domain(self, domain: Literal["sic", "soc"]) -> None:
        if self.classification_type != domain:
            raise AttributeError(
                f"{domain} method is not available for classification_type="
                f"{self.classification_type!r}"
            )

    @staticmethod
    def _create_chat_model(  # noqa: PLR0913
        model_name: str,
        model_location: str,
        llm: ChatVertexAI | ChatOpenAI | None,
        max_tokens: int,
        temperature: float,
        openai_api_key: SecretStr | None,
    ) -> ChatVertexAI | ChatOpenAI:
        if llm is not None:
            return llm
        if model_name.startswith("text-") or model_name.startswith("gemini"):
            return ChatVertexAI(
                model_name=model_name,
                max_output_tokens=max_tokens,
                temperature=temperature,
                location=model_location,
                model_kwargs={"thinking_budget": 0},
            )
        if model_name.startswith("gpt"):
            if openai_api_key is None:
                raise NotImplementedError("Need to provide an OpenAI API key")
            return ChatOpenAI(
                model=model_name,
                api_key=openai_api_key,
                temperature=temperature,
                model_kwargs={"max_tokens": max_tokens},
            )
        raise NotImplementedError("Unsupported model family")

    @staticmethod
    def _coerce_unknown(value: str | None) -> str:
        return "Unknown" if value is None or value in {"", " "} else value

    def _prompt_candidate(
        self,
        code: str,
        examples: list[str],
        include_all: bool = False,
    ) -> str:
        """Format one candidate for the active classification domain."""
        if self.classification_type == "soc":
            return self._prompt_candidate_soc(code, examples, include_all=include_all)
        return self._prompt_candidate_sic(code, examples, include_all=include_all)

    def _prompt_candidate_list(  # noqa: PLR0913
        self,
        short_list: list[dict],
        chars_limit: int = 14000,
        candidates_limit: int | None = None,
        examples_limit: int = 3,
        code_digits: int | None = None,
        *,
        activities_limit: int | None = None,
        titles_limit: int | None = None,
    ) -> str:
        """Create candidate list for the prompt based on the given parameters."""
        if candidates_limit is None:
            candidates_limit = self.config["llm"]["candidates_limit"]
        if code_digits is None:
            code_digits = self.config["llm"]["code_digits"]
        if activities_limit is not None:
            examples_limit = activities_limit
        if titles_limit is not None:
            examples_limit = titles_limit

        grouped: defaultdict[Any, list] = defaultdict(list)
        logger.debug(
            f"Chars Lmt: {chars_limit} Candidate Lmt: {candidates_limit} "
            f"Examples Lmt: {examples_limit} Short List Len: {len(short_list)} "
            f"Code Digits: {code_digits}"
        )
        for item in short_list:
            if item["title"] not in grouped[item["code"][:code_digits]]:
                grouped[item["code"][:code_digits]].append(item["title"])

        candidates = [
            self._prompt_candidate(code, examples[:examples_limit])
            for code, examples in grouped.items()
        ][:candidates_limit]

        if chars_limit:
            chars_count = np.cumsum([len(x) for x in candidates])
            nn = sum(x <= chars_limit for x in chars_count)
            if nn < len(candidates):
                logger.warning(
                    f"Shortening list of candidates to fit token limit from "
                    f"{len(candidates)} to {nn}"
                )
                candidates = candidates[:nn]

        return "\n".join(candidates)

    async def formulate_open_question(
        self,
        industry_descr: str,
        job_title: str | None = None,
        job_description: str | None = None,
        llm_output: SicCandidate | RagCandidate | list | None = None,
        correlation_id: str | None = None,
    ) -> tuple[OpenFollowUp, Any]:
        """Formulate an open-ended follow-up question for the active domain."""
        prompt = (
            self.soc_prompt_openfollowup
            if self.classification_type == "soc"
            else self.sic_prompt_openfollowup
        )
        call_dict = {
            "industry_descr": industry_descr,
            "job_title": self._coerce_unknown(job_title),
            "job_description": self._coerce_unknown(job_description),
            "llm_output": str(llm_output),
        }

        if self.verbose:
            final_prompt = prompt.format(**call_dict)
            logger.debug(final_prompt)

        chain = prompt | self.llm
        logger.info(
            "LLM request sent - formulate_open_question",
            job_title=truncate_identifier(job_title),
            job_description=truncate_identifier(job_description),
            industry_descr=truncate_identifier(industry_descr),
            correlation_id=correlation_id or "",
        )
        llm_start = time.perf_counter()

        try:
            response = await chain.ainvoke(call_dict, return_only_outputs=True)
        except (ValueError, AttributeError) as err:
            logger.error(
                f"Error from LLMChain, exit early: {err}",
                error=str(err),
                correlation_id=correlation_id or "",
            )
            logger.warning(
                "Error from LLMChain, exit early",
                correlation_id=correlation_id or "",
            )
            validated_answer = OpenFollowUp(
                followup=None,
                reasoning="Error from LLMChain, exit early",
            )
            return validated_answer, call_dict

        llm_duration_ms = int((time.perf_counter() - llm_start) * 1000)
        parser = PydanticOutputParser(pydantic_object=OpenFollowUp)
        try:
            validated_answer = parser.parse(str(response.content))
            has_followup = bool(getattr(validated_answer, "followup", None))
            logger.info(
                "LLM response received for open question prompt",
                has_followup=str(has_followup),
                duration_ms=str(llm_duration_ms),
                correlation_id=correlation_id or "",
            )
        except (ValueError, AttributeError) as parse_error:
            logger.error(
                f"Failed to parse response: {parse_error}",
                error=str(parse_error),
                correlation_id=correlation_id or "",
            )
            logger.warning(
                "Failed to parse response",
                response_content=str(response.content),
                correlation_id=correlation_id or "",
            )
            logger.info(
                "LLM response received for open question prompt",
                has_followup="False",
                duration_ms=str(llm_duration_ms),
                correlation_id=correlation_id or "",
            )
            try:
                chain = FIX_PARSING_PROMPT | self.llm
                response = await chain.ainvoke(
                    {
                        "llm_output": str(response.content),
                        "format_instructions": parser.get_format_instructions(),
                    },
                    return_only_outputs=True,
                )
                validated_answer = parser.parse(str(response.content))
                logger.debug("Successfully parsed reformatted response.")
            except (ValueError, AttributeError) as parse_error2:
                logger.error(
                    f"Failed to parse response again: {parse_error2}",
                    error=str(parse_error2),
                )
                logger.warning(
                    "Failed to parse response again",
                    response_content=str(response.content),
                )
                reasoning = (
                    f"ERROR parse_error=<{parse_error2}>, response=<{response.content}>"
                )
                validated_answer = OpenFollowUp(
                    followup=None,
                    reasoning=reasoning,
                )

        if self.verbose:
            logger.debug(f"{response=}")

        return validated_answer, call_dict

    def _prompt_candidate_sic(
        self, code: str, activities: list[str], include_all: bool = False
    ) -> str:
        """Reformat the candidate activities for the prompt.

        Args:
            code (str): The code for the item.
            activities (list[str]): The list of example activities.
            include_all (bool, optional): Whether to include all the sic metadata.

        Returns:
            str: A formatted string containing the code, title, and example activities.
        """
        self._require_domain("sic")
        if self.sic is None:
            self.sic = load_sic_hierarchy(
                self.config["lookups"]["sic_index"],
                self.config["lookups"]["sic_structure"],
            )

        item = self.sic[code]
        txt = "{" + f"Code: {item.numeric_string_padded()}, Title: {item.description}"
        txt += f", Example activities: {', '.join(activities)}"
        if include_all:
            if item.sic_meta.detail:
                txt += f", Details: {item.sic_meta.detail}"
            if item.sic_meta.includes:
                txt += f", Includes: {', '.join(item.sic_meta.includes)}"
            if item.sic_meta.excludes:
                txt += f", Excludes: {', '.join(item.sic_meta.excludes)}"
        return txt + "}"

    def _prompt_candidate_soc(
        self,
        code: str,
        job_titles: list[str],
        include_all: bool = False,
    ) -> str:
        """Reformat the candidate activities for the prompt.

        Args:
            code (str): The code for the item.
            job_titles (list[str]): The list of example job titles.
            include_all (bool, optional): Whether to include all the soc metadata.

        Returns:
            str: A formatted string containing the code, title, and example activities.
        """
        self._require_domain("soc")
        if self.soc is None:
            self.soc = load_soc_hierarchy(
                self.config["lookups"]["soc_index"],
                self.config["lookups"]["soc_structure"],
            )

        item = self.soc[code]
        txt = "{" + f"Code: {item.soc_code}, Title: {item.group_title}"
        txt += f", Example job_titles: {', '.join(job_titles)}"
        if include_all:
            if item.group_description:
                txt += f", Details: {item.group_description}"
            tasks = item.tasks or self.soc_meta.get(code, {}).get("tasks") or []
            if tasks:
                txt += f", Includes: {', '.join(tasks)}"
        return txt + "}"

    @lru_cache  # noqa: B019
    async def get_sic_code(
        self,
        industry_descr: str,
        job_title: str,
        job_description: str,
    ) -> SicResponse:
        """Generates a SIC classification based on respondent's data
        using a whole condensed index embedded in the query.

        Args:
            industry_descr (str): Description of the industry.
            job_title (str): Title of the job.
            job_description (str): Description of the job.

        Returns:
            SicResponse: Generated response to the query.
        """
        self._require_domain("sic")
        chain = self.sic_prompt | self.llm
        response = await chain.ainvoke(
            {
                "industry_descr": industry_descr,
                "job_title": job_title,
                "job_description": job_description,
            },
            return_only_outputs=True,
        )
        if self.verbose:
            logger.debug(f"LLM response: {response}")
        # Parse the output to desired format with one retry
        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=SicResponse
        )
        try:
            validated_answer = parser.parse(str(response.content))
        except ValueError as parse_error:
            logger.debug(
                f"Retrying llm response parsing due to an error: {parse_error}"
            )
            logger.error(f"Unable to parse llm response: {parse_error}")

            reasoning = (
                f"ERROR parse_error=<{parse_error}>, response=<{response.content}>"
            )
            validated_answer = SicResponse(
                codable=False,
                sic_candidates=[],
                reasoning=reasoning,
            )

        return validated_answer

    def _prompt_candidate_list_filtered(  # noqa: PLR0913
        self,
        short_list: list[dict],
        chars_limit: int = 14000,
        candidates_limit: int | None = None,
        activities_limit: int = 3,
        code_digits: int | None = None,
        filtered_list: list[str] | None = None,
    ) -> str:
        """Create candidate list for the prompt based on the given parameters.

        This method takes a structured list of candidates and generates a short
        string list based on the provided parameters. It filters the candidates
        based on the code digits and activities limit, and shortens the list to
        fit the character limit.

        Args:
            short_list (list[dict]): A list of candidate dictionaries.
            chars_limit (int, optional): The character limit for the generated
                prompt. Defaults to 14000.
            candidates_limit (int, optional): The maximum number of candidates
                to include in the prompt. Defaults to 5.
            activities_limit (int, optional): The maximum number of activities
                to include for each code. Defaults to 3.
            code_digits (int, optional): The number of digits to consider from
                the code for filtering candidates. Defaults to 5.
            filtered_list (list[str], optional): A list of alternative
                candidates.

        Returns:
            str: The generated candidate list for the prompt.
        """
        self._require_domain("sic")
        if candidates_limit is None:
            candidates_limit = self.config["llm"]["candidates_limit"]
        if code_digits is None:
            code_digits = self.config["llm"]["code_digits"]
        if not filtered_list:
            logger.warning("Empty list")
            return ""

        a: defaultdict[Any, list] = defaultdict(list)
        for item in short_list:
            if (
                item["code"] in filtered_list
                and item["title"] not in a[item["code"][:code_digits]]
            ):
                a[item["code"][:code_digits]].append(item["title"])

            sic_candidates = [
                self._prompt_candidate(code, activities[:activities_limit])
                for code, activities in a.items()
            ][:candidates_limit]

        if chars_limit:
            chars_count = np.cumsum([len(x) for x in sic_candidates])
            nn = sum(x <= chars_limit for x in chars_count)
            if nn < len(sic_candidates):
                logger.warning(  # pylint: disable=logging-not-lazy
                    "Shortening list of candidates to fit token limit "
                    + f"from {len(sic_candidates)} to {nn}"
                )
                sic_candidates = sic_candidates[:nn]

        return "\n".join(sic_candidates)

    async def sa_rag_sic_code(  # noqa: PLR0913
        self,
        industry_descr: str,
        job_title: str | None = None,
        job_description: str | None = None,
        code_digits: int | None = None,
        candidates_limit: int | None = None,
        short_list: list[dict[Any, Any]] | None = None,
    ) -> tuple[SicResponse, list[dict[Any, Any]] | None, Any | None]:
        """Generates a SIC classification based on respondent's data using RAG approach.

        Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            code_digits (int, optional): The number of digits in the generated
                SIC code. Defaults to 5.
            candidates_limit (int, optional): The maximum number of SIC code candidates
                to consider. Defaults to 5.
            short_list (list[dict[Any, Any]], optional): A list of results from embedding search

        Returns:
            SicResponse: The generated response to the query.

        Raises:
            ValueError: If there is an error during the parsing of the response.
            ValueError: If the default embedding handler is required but
                not loaded correctly.

        """
        self._require_domain("sic")
        if candidates_limit is None:
            candidates_limit = self.config["llm"]["candidates_limit"]
        if code_digits is None:
            code_digits = self.config["llm"]["code_digits"]

        def prep_call_dict(industry_descr, job_title, job_description, sic_codes):
            # Helper function to prepare the call dictionary
            is_job_title_present = job_title is None or job_title in {"", " "}
            job_title = "Unknown" if is_job_title_present else job_title

            is_job_description_present = job_description is None or job_description in {
                "",
                " ",
            }
            job_description = (
                "Unknown" if is_job_description_present else job_description
            )

            call_dict = {
                "industry_descr": industry_descr,
                "job_title": job_title,
                "job_description": job_description,
                "sic_index": sic_codes,
            }
            return call_dict

        if short_list is None:
            raise ValueError(
                "Short list is None - list provided from embedding search."
            )

        sic_codes = self._prompt_candidate_list(
            short_list, code_digits=code_digits, candidates_limit=candidates_limit
        )

        call_dict = prep_call_dict(
            industry_descr=industry_descr,
            job_title=job_title,
            job_description=job_description,
            sic_codes=sic_codes,
        )

        if self.verbose:
            final_prompt = self.sa_sic_prompt_rag.format(**call_dict)
            logger.debug(f"Final prompt: {final_prompt}")

        chain = self.sa_sic_prompt_rag | self.llm

        try:
            response = await chain.ainvoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.error(f"Error from chain, exit early: {err}", error=str(err))
            validated_answer = SicResponse(
                followup="Follow-up question not available due to error.",
                reasoning="Error from chain, exit early",
            )
            return validated_answer, short_list, call_dict
        if self.verbose:
            logger.debug(f"LLM response: {response}")

        # Parse the output to the desired format
        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=SicResponse
        )
        try:
            validated_answer = parser.parse(str(response.content))
        except (ValueError, AttributeError) as parse_error:
            logger.error(
                f"Failed to parse response: {parse_error}", error=str(parse_error)
            )
            logger.warning(
                "Failed to parse response", response_content=str(response.content)
            )

            # send another llm request to fix the format (1 attempt)
            try:
                chain = FIX_PARSING_PROMPT | self.llm
                response = await chain.ainvoke(
                    {
                        "llm_output": str(response.content),
                        "format_instructions": parser.get_format_instructions(),
                    },
                    return_only_outputs=True,
                )
                validated_answer = parser.parse(str(response.content))
                logger.debug("Successfully parsed reformatted response.")
            except (ValueError, AttributeError) as parse_error2:
                logger.error(
                    f"Failed to parse response again: {parse_error2}",
                    error=str(parse_error2),
                )
                logger.warning(
                    "Failed to parse response again",
                    response_content=str(response.content),
                )
                reasoning = (
                    f"ERROR parse_error=<{parse_error2}>, response=<{response.content}>"
                )
                validated_answer = SicResponse(
                    followup="Follow-up question not available due to error.",
                    reasoning=reasoning,
                )

        return validated_answer, short_list, call_dict

    async def unambiguous_sic_code(  # noqa: PLR0913
        self,
        industry_descr: str,
        semantic_search_results: list[dict],
        job_title: str | None = None,
        job_description: str | None = None,
        candidates_limit: int | None = None,
        code_digits: int | None = None,
        correlation_id: str | None = None,
    ) -> tuple[UnambiguousResponse, Any | None]:
        """Evaluates codability to a single 5-digit SIC code based on respondent's data.

        Args:
            industry_descr (str): The description of the industry.
            semantic_search_results (list of dicts): List of semantic search results.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            candidates_limit (int, optional): The maximum number of candidates
                to include in the prompt. Defaults to 5.
            code_digits (int, optional): The number of digits to consider from
                the code for filtering candidates. Defaults to 5.
            correlation_id (str, optional): Optional correlation ID for request tracking.

        Returns:
            UnambiguousResponse: The generated response to the query.

        Raises:
            ValueError: If there is an error during the parsing of the response.
            ValueError: If the default embedding handler is required but
                not loaded correctly.

        """
        self._require_domain("sic")
        if candidates_limit is None:
            candidates_limit = self.config["llm"]["candidates_limit"]
        if code_digits is None:
            code_digits = self.config["llm"]["code_digits"]
        sic_candidates = self._prompt_candidate_list(
            short_list=semantic_search_results,
            code_digits=code_digits,
            candidates_limit=candidates_limit,
        )

        job_title = (
            "Unknown" if (job_title is None or job_title in {"", " "}) else job_title
        )
        job_description = (
            "Unknown"
            if (job_description is None or job_description in {"", " "})
            else job_description
        )

        call_dict = {
            "industry_descr": industry_descr,
            "job_title": job_title,
            "job_description": job_description,
            "sic_candidates": sic_candidates,
        }

        if self.verbose:
            final_prompt = self.sic_prompt_unambiguous.format(**call_dict)
            logger.debug(final_prompt)

        chain = self.sic_prompt_unambiguous | self.llm

        # Log LLM request sent
        logger.info(
            "LLM request sent - unambiguous_sic_code",
            job_title=truncate_identifier(job_title),
            job_description=truncate_identifier(job_description),
            industry_descr=truncate_identifier(industry_descr),
            correlation_id=correlation_id or "",
        )
        llm_start = time.perf_counter()

        try:
            response = await chain.ainvoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.error(
                f"Error from chain, exit early: {err}",
                error=str(err),
                correlation_id=correlation_id or "",
            )
            validated_answer = UnambiguousResponse(
                codable=False,
                alt_candidates=[],
                reasoning="Error from chain, exit early",
            )
            return validated_answer, call_dict

        if self.verbose:
            logger.debug(f"llm_response={response}")

        # Parse the output to the desired format
        parser = PydanticOutputParser(pydantic_object=UnambiguousResponse)  # type: ignore
        try:
            validated_answer = parser.parse(str(response.content))
            # Log LLM response received after successful parse
            alt_candidates_count = len(
                getattr(validated_answer, "alt_candidates", []) or []
            )
            codable = bool(getattr(validated_answer, "codable", False))
            selected_code = (
                str(getattr(validated_answer, "class_code", "")) if codable else ""
            )
            llm_duration_ms = int((time.perf_counter() - llm_start) * 1000)
            logger.info(
                "LLM response received for unambiguous sic prompt",
                codable=str(codable),
                selected_code=selected_code,
                alt_candidates_count=str(alt_candidates_count),
                duration_ms=str(llm_duration_ms),
                correlation_id=correlation_id or "",
            )
        except (ValueError, AttributeError) as parse_error:
            logger.error(
                f"Failed to parse response: {parse_error}",
                error=str(parse_error),
                correlation_id=correlation_id or "",
            )
            llm_duration_ms = int((time.perf_counter() - llm_start) * 1000)
            logger.warning(
                "Failed to parse response",
                response_content=str(response.content),
                duration_ms=str(llm_duration_ms),
                correlation_id=correlation_id or "",
            )

            # send another llm request to fix the format (1 attempt)
            try:
                chain = FIX_PARSING_PROMPT | self.llm
                response = await chain.ainvoke(
                    {
                        "llm_output": str(response.content),
                        "format_instructions": parser.get_format_instructions(),
                    },
                    return_only_outputs=True,
                )
                validated_answer = parser.parse(str(response.content))
                logger.debug("Successfully parsed reformatted response.")

            except (ValueError, AttributeError) as parse_error2:
                logger.error(
                    f"Failed to parse response again: {parse_error2}",
                    error=str(parse_error2),
                )
                logger.warning(
                    "Failed to parse response again",
                    response_content=str(response.content),
                )
                reasoning = (
                    f"ERROR parse_error=<{parse_error2}>, response=<{response.content}>"
                )
                validated_answer = UnambiguousResponse(
                    codable=False,
                    alt_candidates=[],
                    reasoning=reasoning,
                )

        return validated_answer, call_dict

    async def reranker_sic(  # noqa: PLR0913
        self,
        industry_descr: str,
        job_title: str | None = None,
        job_description: str | None = None,
        code_digits: int | None = None,
        candidates_limit: int | None = None,
        output_limit: int = 5,
        short_list: list[dict[Any, Any]] | None = None,
    ) -> tuple[Any, list | None, dict[str, Any] | None] | dict[str, Any]:
        """Generates a set of relevant SIC codes based on respondent's data
            using reranking approach.

        Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            code_digits (int, optional): The number of digits in the generated
                SIC code. Defaults to 5.
            candidates_limit (int, optional): The maximum number of SIC code candidates
                to consider. Defaults to 7.
            output_limit (int, optional): The maximum number of SIC codes to return.
                Defaults to 5.
            short_list (list[dict[Any, Any]], optional): A list of results from embedding search.

        Returns:
            tuple[RerankingResponse, dict[str, Any]]: The reranking response and additional data.

        Raises:
            ValueError: If there is an error during the parsing of the response.
            ValueError: If the default embedding handler is required but
                not loaded correctly.

        """
        self._require_domain("sic")
        if candidates_limit is None:
            candidates_limit = self.config["llm"]["candidates_limit"]
        if code_digits is None:
            code_digits = self.config["llm"]["code_digits"]

        def prep_call_dict(
            industry_descr, job_title, job_description, sic_codes, output_limit
        ):
            # Helper function to prepare the call dictionary
            is_job_title_present = job_title is None or job_title in {"", " "}
            job_title = "Unknown" if is_job_title_present else job_title

            is_job_description_present = job_description is None or job_description in {
                "",
                " ",
            }
            job_description = (
                "Unknown" if is_job_description_present else job_description
            )

            call_dict = {
                "industry_descr": industry_descr,
                "job_title": job_title,
                "job_description": job_description,
                "sic_index": sic_codes,
                "n": output_limit,
            }
            return call_dict

        if short_list is None:
            raise ValueError(
                "Short list is None - list provided from embedding search."
            )

        sic_codes = self._prompt_candidate_list(
            short_list, code_digits=code_digits, candidates_limit=candidates_limit
        )

        call_dict = prep_call_dict(
            industry_descr=industry_descr,
            job_title=job_title,
            job_description=job_description,
            sic_codes=sic_codes,
            output_limit=output_limit,
        )

        if self.verbose:
            final_prompt = self.sic_prompt_reranker.format(**call_dict)
            logger.debug(f"Final prompt: {final_prompt}")

        chain = self.sic_prompt_reranker | self.llm

        try:
            response = await chain.ainvoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.error(f"Error from chain, exit early: {err}", error=str(err))
            validated_answer = RerankingResponse(
                selected_codes=[],
                excluded_codes=[],
                status="Error from chain, exit early",
                n_requested=output_limit,
            )
            return validated_answer, short_list, call_dict

        if self.verbose:
            logger.debug(f"LLM response: {response}")

        # Parse the output to the desired format
        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=RerankingResponse
        )
        try:
            validated_answer = parser.parse(str(response.content))
        except ValueError as parse_error:
            logger.error(
                f"Failed to parse response: {parse_error}", error=str(parse_error)
            )
            logger.warning(
                "Failed to parse response", response_content=str(response.content)
            )

            reasoning = (
                f"ERROR parse_error=<{parse_error}>, response=<{response.content}>"
            )
            validated_answer = RerankingResponse(
                selected_codes=[],
                excluded_codes=[],
                status=reasoning,
                n_requested=output_limit,
            )

        return validated_answer, short_list, call_dict

    async def final_sic_code(  # noqa: PLR0913
        self,
        industry_descr: str,
        job_title: str | None = None,
        job_description: str | None = None,
        sic_candidates: str | None = None,
        open_question: str | None = None,
        answer_to_open_question: str | None = None,
        closed_question: str | None = None,
        answer_to_closed_question: str | None = None,
    ) -> tuple[FinalSICAssignment, Any | None]:
        """Evaluates codability to a single 5-digit SIC code based on respondent's data
            and answers to follow-up questions.

        Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            sic_candidates: (str, optional): Short list of SIC candidates to pass to LLM.
            open_question (str, optional): The open question. Defaults to None.
            answer_to_open_question (str, optional): The answer to the open question.
                Defaults to None.
            closed_question (str, optional): The closed question. Defaults to None.
            answer_to_closed_question (str, optional): The answer to the closed question.
                Defaults to None.

        Returns:
            FinalSICAssignment: The generated response to the query.

        Raises:
            ValueError: If there is an error during the parsing of the response.
            ValueError: If the default embedding handler is required but
                not loaded correctly.

        """
        self._require_domain("sic")

        def prep_call_dict(  # noqa: PLR0913
            industry_descr,
            job_title,
            job_description,
            sic_candidates,
            open_question,
            answer_to_open_question,
            closed_question,
            answer_to_closed_question,
        ):
            # Helper function to prepare the call dictionary
            is_job_title_present = job_title is None or job_title in {"", " "}
            job_title = "Unknown" if is_job_title_present else job_title

            is_job_description_present = job_description is None or job_description in {
                "",
                " ",
            }
            job_description = (
                "Unknown" if is_job_description_present else job_description
            )

            call_dict = {
                "industry_descr": industry_descr,
                "job_title": job_title,
                "job_description": job_description,
                "sic_candidates": sic_candidates,
                "open_question": open_question,
                "answer_to_open_question": answer_to_open_question,
                "closed_question": closed_question,
                "answer_to_closed_question": answer_to_closed_question,
            }
            return call_dict

        call_dict = prep_call_dict(
            industry_descr=industry_descr,
            job_title=job_title,
            job_description=job_description,
            sic_candidates=sic_candidates,
            open_question=open_question,
            answer_to_open_question=answer_to_open_question,
            closed_question=closed_question,
            answer_to_closed_question=answer_to_closed_question,
        )

        if self.verbose:
            final_prompt = self.sic_prompt_final.format(**call_dict)
            logger.debug(f"Final prompt: {final_prompt}")

        chain = self.sic_prompt_final | self.llm

        try:
            response = await chain.ainvoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.error(f"Error from chain, exit early: {err}", error=str(err))
            validated_answer = FinalSICAssignment(
                codable=False,
                unambiguous_code="N/A",
                unambiguous_code_descriptive="N/A",
                higher_level_code="N/A",
                reasoning="Error from chain, exit early",
            )
            return validated_answer, call_dict

        if self.verbose:
            logger.debug(f"llm_response={response}")

        # Parse the output to the desired format
        parser = PydanticOutputParser(pydantic_object=FinalSICAssignment)  # type: ignore
        try:
            validated_answer = parser.parse(str(response.content))
        except (ValueError, AttributeError) as parse_error:
            logger.error(
                f"Failed to parse response: {parse_error}", error=str(parse_error)
            )
            logger.warning(
                "Failed to parse response", response_content=str(response.content)
            )

            try:
                chain = FIX_PARSING_PROMPT | self.llm
                response = await chain.ainvoke(
                    {
                        "llm_output": str(response.content),
                        "format_instructions": parser.get_format_instructions(),
                    },
                    return_only_outputs=True,
                )
                validated_answer = parser.parse(str(response.content))
                logger.debug("Successfully parsed reformatted response.")

            except (ValueError, AttributeError) as parse_error2:
                logger.error(
                    f"Failed to parse response again: {parse_error2}",
                    error=str(parse_error2),
                )
                logger.warning(
                    "Failed to parse response again",
                    response_content=str(response.content),
                )
                reasoning = (
                    f"ERROR parse_error=<{parse_error2}>, response=<{response.content}>"
                )
                validated_answer = FinalSICAssignment(
                    codable=False,
                    unambiguous_code="N/A",
                    unambiguous_code_descriptive="N/A",
                    higher_level_code="N/A",
                    reasoning=reasoning,
                )

        return validated_answer, call_dict

    async def formulate_closed_question(
        self,
        industry_descr: str,
        job_title: str | None = None,
        job_description: str | None = None,
        llm_output: UnambiguousResponse | None = None,
        correlation_id: str | None = None,
    ) -> tuple[ClosedFollowUp, Any]:
        """Formulates a closed follow-up question using respondent data
            and survey design guidelines.

        Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            llm_output (UnambiguousResponse, optional): The response from the LLM model.
            correlation_id (str, optional): Optional correlation ID for request tracking.

        Returns:
            ClosedFollowUp: The generated response to the query.

        Raises:
            ValueError: If there is an error during the parsing of the response.
            ValueError: If the default embedding handler is required but
                not loaded correctly.

        """
        self._require_domain("sic")

        def prep_call_dict(industry_descr, job_title, job_description, llm_output):
            # Helper function to prepare the call dictionary
            is_job_title_present = job_title is None or job_title in {"", " "}
            job_title = "Unknown" if is_job_title_present else job_title

            is_job_description_present = job_description is None or job_description in {
                "",
                " ",
            }
            job_description = (
                "Unknown" if is_job_description_present else job_description
            )

            call_dict = {
                "industry_descr": industry_descr,
                "job_title": job_title,
                "job_description": job_description,
                "llm_output": str(llm_output),
            }
            return call_dict

        call_dict = prep_call_dict(
            industry_descr=industry_descr,
            job_title=job_title,
            job_description=job_description,
            llm_output=llm_output,
        )

        if self.verbose:
            final_prompt = self.sic_prompt_closedfollowup.format(**call_dict)
            logger.debug(f"Final prompt: {final_prompt}")

        chain = self.sic_prompt_closedfollowup | self.llm

        try:
            response = await chain.ainvoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.error(
                f"Error from LLMChain, exit early: {err}",
                error=str(err),
                correlation_id=correlation_id or "",
            )
            logger.warning(
                "Error from LLMChain, exit early",
                correlation_id=correlation_id or "",
            )
            validated_answer = ClosedFollowUp(
                followup=None,
                sic_options=[],
                reasoning="Error from LLMChain, exit early",
            )
            return validated_answer, call_dict

        if self.verbose:
            logger.debug(f"{response=}")

        # Parse the output to the desired format
        parser = PydanticOutputParser(pydantic_object=ClosedFollowUp)
        try:
            validated_answer = parser.parse(str(response.content))
        except ValueError as parse_error:
            logger.error(
                f"Failed to parse response: {parse_error}",
                error=str(parse_error),
                correlation_id=correlation_id or "",
            )
            logger.warning(
                "Failed to parse response",
                response_content=str(response.content),
                correlation_id=correlation_id or "",
            )

            reasoning = (
                f"ERROR parse_error=<{parse_error}>, response=<{response.content}>"
            )
            validated_answer = ClosedFollowUp(
                followup=None,
                sic_options=[],
                reasoning=reasoning,
            )

        return validated_answer, call_dict

    @lru_cache  # noqa: B019
    async def get_soc_code(
        self,
        job_title: str,
        job_description: str,
        level_of_education: str,
        manage_others: bool,
        industry_descr: str,
    ) -> SocResponse:
        """Generates a SOC classification based on respondent's data
        using the full SOC index embedded in the query (mirror SIC one-shot).

        Args:
            job_title (str): The title of the job.
            job_description (str): The description of the job.
            level_of_education (str): The level of education required for the job.
            manage_others (bool): Indicates whether the job involves managing others.
            industry_descr (str): The description of the industry.

        Returns:
            SocResponse: The generated response to the query.

        Raises:
            ValueError: If there is an error parsing the response from the LLM model.

        """
        self._require_domain("soc")
        chain = self.soc_prompt | self.llm
        response = await chain.ainvoke(
            {
                "job_title": job_title,
                "job_description": job_description,
                "level_of_education": level_of_education,
                "manage_others": manage_others,
                "industry_descr": industry_descr,
            },
            return_only_outputs=True,
        )
        if self.verbose:
            logger.debug(f"LLM response: {response}")
        # Parse the output to desired format with one retry
        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=SocResponse,
        )
        try:
            validated_answer_sr = parser.parse(str(response.content))
        except ValueError as parse_error:
            logger.error(f"Unable to parse llm response: {parse_error}")
            content = getattr(response, "content", "")
            reasoning = f"ERROR parse_error=<{parse_error}>, response=<{content}>"
            validated_answer_sr = SocResponse(
                codable=False, soc_candidates=[], reasoning=reasoning
            )

        return validated_answer_sr

    async def unambiguous_soc_code(  # noqa: PLR0913
        self,
        industry_descr: str,
        semantic_search_results: list[dict],
        job_title: str | None = None,
        job_description: str | None = None,
        candidates_limit: int | None = None,
        code_digits: int | None = None,
        correlation_id: str | None = None,
    ) -> tuple[UnambiguousResponse, dict[str, Any]]:
        """Evaluate codability to a single four-digit SOC unit group (mirrors SIC)."""
        self._require_domain("soc")
        if candidates_limit is None:
            candidates_limit = self.config["llm"]["candidates_limit"]
        if code_digits is None:
            code_digits = self.config["llm"]["code_digits"]
        soc_candidates = self._prompt_candidate_list(
            short_list=semantic_search_results,
            code_digits=code_digits,
            candidates_limit=candidates_limit,
        )

        job_title = (
            "Unknown" if (job_title is None or job_title in {"", " "}) else job_title
        )
        job_description = (
            "Unknown"
            if (job_description is None or job_description in {"", " "})
            else job_description
        )

        call_dict = {
            "industry_descr": industry_descr,
            "job_title": job_title,
            "job_description": job_description,
            "soc_candidates": soc_candidates,
        }

        if self.verbose:
            final_prompt = self.soc_prompt_unambiguous.format(**call_dict)
            logger.debug(final_prompt)

        chain = self.soc_prompt_unambiguous | self.llm
        logger.info(
            "LLM request sent - unambiguous_soc_code",
            job_title=truncate_identifier(job_title),
            job_description=truncate_identifier(job_description),
            industry_descr=truncate_identifier(industry_descr),
            correlation_id=correlation_id or "",
        )
        llm_start = time.perf_counter()

        try:
            response = await chain.ainvoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.error(
                f"Error from chain, exit early: {err}",
                error=str(err),
                correlation_id=correlation_id or "",
            )
            validated_answer = UnambiguousResponse(
                codable=False,
                alt_candidates=[],
                reasoning="Error from chain, exit early",
            )
            return validated_answer, call_dict

        if self.verbose:
            logger.debug(f"llm_response={response}")

        parser = PydanticOutputParser(pydantic_object=UnambiguousResponse)  # type: ignore
        try:
            validated_answer = parser.parse(str(response.content))
            alt_candidates_count = len(
                getattr(validated_answer, "alt_candidates", []) or []
            )
            codable = bool(getattr(validated_answer, "codable", False))
            selected_code = (
                str(getattr(validated_answer, "class_code", "")) if codable else ""
            )
            llm_duration_ms = int((time.perf_counter() - llm_start) * 1000)
            logger.info(
                "LLM response received for unambiguous soc prompt",
                codable=str(codable),
                selected_code=selected_code,
                alt_candidates_count=str(alt_candidates_count),
                duration_ms=str(llm_duration_ms),
                correlation_id=correlation_id or "",
            )
        except (ValueError, AttributeError) as parse_error:
            logger.error(
                f"Failed to parse response: {parse_error}",
                error=str(parse_error),
                correlation_id=correlation_id or "",
            )
            llm_duration_ms = int((time.perf_counter() - llm_start) * 1000)
            logger.warning(
                "Failed to parse response",
                response_content=str(response.content),
                duration_ms=str(llm_duration_ms),
                correlation_id=correlation_id or "",
            )
            try:
                fix_chain = FIX_PARSING_PROMPT | self.llm
                response = await fix_chain.ainvoke(
                    {
                        "llm_output": str(response.content),
                        "format_instructions": parser.get_format_instructions(),
                    },
                    return_only_outputs=True,
                )
                validated_answer = parser.parse(str(response.content))
                logger.debug("Successfully parsed reformatted response.")
            except (ValueError, AttributeError) as parse_error2:
                logger.error(
                    f"Failed to parse response again: {parse_error2}",
                    error=str(parse_error2),
                )
                logger.warning(
                    "Failed to parse response again",
                    response_content=str(response.content),
                )
                reasoning = (
                    f"ERROR parse_error=<{parse_error2}>, response=<{response.content}>"
                )
                validated_answer = UnambiguousResponse(
                    codable=False,
                    alt_candidates=[],
                    reasoning=reasoning,
                )

        return validated_answer, call_dict

    async def sa_rag_soc_code(  # noqa: PLR0913
        self,
        industry_descr: str,
        job_title: str | None = None,
        job_description: str | None = None,
        code_digits: int | None = None,
        candidates_limit: int | None = None,
        short_list: list[dict[Any, Any]] | None = None,
    ) -> tuple[SocResponse, list[dict[Any, Any]] | None, Any | None]:
        """Generates a SOC classification based on respondent's data using RAG approach.

        Caller must provide short_list (e.g. from vector store API). Mirrors
        sic-classification-utils ``sa_rag_sic_code`` (raises when short_list is None).

        Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            code_digits (int, optional): The number of digits in the generated
                SOC code. Defaults to 4.
            candidates_limit (int, optional): The maximum number of SOC code candidates
                to consider. Defaults to 5.
            short_list (list[dict[Any, Any]], optional): A list of results from
                embedding or vector store search (e.g. from soc-classification-vector-store).
                Each dict should have "code" and "title" keys.

        Returns:
            SocResponse: The generated response to the query.

        Raises:
            ValueError: If there is an error during the parsing of the response.
            ValueError: If short_list is None.

        """
        self._require_domain("soc")
        if candidates_limit is None:
            candidates_limit = self.config["llm"]["candidates_limit"]
        if code_digits is None:
            code_digits = self.config["llm"]["code_digits"]

        def prep_call_dict(industry_descr, job_title, job_description, soc_codes):
            # Helper function to prepare the call dictionary
            is_job_title_present = job_title is None or job_title in {"", " "}
            job_title = "Unknown" if is_job_title_present else job_title

            is_job_description_present = job_description is None or job_description in {
                "",
                " ",
            }
            job_description = (
                "Unknown" if is_job_description_present else job_description
            )

            call_dict = {
                "industry_descr": industry_descr,
                "job_title": job_title,
                "job_description": job_description,
                "soc_index": soc_codes,
            }
            return call_dict

        if short_list is None:
            raise ValueError(
                "Short list is None - list provided from embedding search."
            )

        soc_codes = self._prompt_candidate_list(
            short_list, code_digits=code_digits, candidates_limit=candidates_limit
        )

        call_dict = prep_call_dict(
            industry_descr=industry_descr,
            job_title=job_title,
            job_description=job_description,
            soc_codes=soc_codes,
        )

        if self.verbose:
            final_prompt = self.sa_soc_prompt_rag.format(**call_dict)
            logger.debug(f"Final prompt: {final_prompt}")

        chain = self.sa_soc_prompt_rag | self.llm

        try:
            response = await chain.ainvoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.error(f"Error from chain, exit early: {err}", error=str(err))
            validated_answer = SocResponse(
                followup="Follow-up question not available due to error.",
                reasoning="Error from chain, exit early",
            )
            return validated_answer, short_list, call_dict

        if self.verbose:
            logger.debug(f"LLM response: {response}")

        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=SocResponse,
        )
        try:
            validated_answer = parser.parse(str(response.content))
        except (ValueError, AttributeError) as parse_error:
            logger.error(
                f"Failed to parse response: {parse_error}", error=str(parse_error)
            )
            logger.warning(
                "Failed to parse response", response_content=str(response.content)
            )

            try:
                chain = FIX_PARSING_PROMPT | self.llm
                response = await chain.ainvoke(
                    {
                        "llm_output": str(response.content),
                        "format_instructions": parser.get_format_instructions(),
                    },
                    return_only_outputs=True,
                )
                validated_answer = parser.parse(str(response.content))
                logger.debug("Successfully parsed reformatted response.")
            except (ValueError, AttributeError) as parse_error2:
                logger.error(
                    f"Failed to parse response again: {parse_error2}",
                    error=str(parse_error2),
                )
                logger.warning(
                    "Failed to parse response again",
                    response_content=str(response.content),
                )
                reasoning = (
                    f"ERROR parse_error=<{parse_error2}>, response=<{response.content}>"
                )
                validated_answer = SocResponse(
                    followup="Follow-up question not available due to error.",
                    reasoning=reasoning,
                )

        return validated_answer, short_list, call_dict
