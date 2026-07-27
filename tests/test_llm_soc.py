# pylint: disable=C0116, R0801, W0621
"""Tests for survey_assist_classification_core.llm.llm (SOC domain)."""

import json
from importlib.resources import as_file, files
from unittest import mock

import pytest
import vertexai
from langchain_core.messages import AIMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from occupational_classification.data_access.soc_data_access import (
    load_soc_index as lib_load_soc_index,
)
from occupational_classification.data_access.soc_data_access import (
    load_soc_structure as lib_load_soc_structure,
)
from occupational_classification.hierarchy.soc_hierarchy import load_hierarchy

from survey_assist_classification_core.llm.llm import ClassificationLLM
from survey_assist_classification_core.models.response_model import (
    OpenFollowUp,
    UnambiguousResponse,
)

MODEL_NAME = "gemini-2.5-flash"
LOCATION = "europe-west2"


# Test initialisation
def test_setup():
    vertexai.init(project="classifai-sandbox", location=LOCATION)


@pytest.fixture(autouse=True)
def mock_vertex_ai():
    with mock.patch(
        "google.cloud.aiplatform.gapic.PredictionServiceClient"
    ) as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.generate_content.return_value = mock.Mock()
        yield


@pytest.mark.parametrize(
    "model, openai_api_key, expected_model",
    [
        ("gemini", None, ChatVertexAI),
        ("text-", None, ChatVertexAI),
        ("gpt", "key", ChatOpenAI),
    ],
)
@pytest.mark.llm
def test_llm_model(model, openai_api_key, expected_model):
    llm_model_type = ClassificationLLM(
        classification_type="soc", model_name=model, openai_api_key=openai_api_key
    ).llm
    assert isinstance(llm_model_type, expected_model)


@pytest.mark.llm
def test_pass_llm_argument():
    llm_model = ClassificationLLM(classification_type="soc", llm="model").llm
    assert llm_model == "model"


@pytest.mark.llm
def test_llm_model_default():
    assert isinstance(ClassificationLLM(classification_type="soc").llm, ChatVertexAI)


@pytest.mark.llm
def test_prompt_candidate_strict_hierarchy_lookup(mock_vertex_ai):
    """Prompt line comes from ``self.soc[code]`` (no vector-store title fallback)."""
    _ = mock_vertex_ai
    llm = ClassificationLLM(classification_type="soc", model_name=MODEL_NAME)
    ref = ("occupational_classification", "data/example_soc_lookup_data.csv")
    with as_file(files(ref[0]).joinpath(ref[1])) as path:
        p = str(path)
        idx = lib_load_soc_index(p)
        llm.soc = load_hierarchy(lib_load_soc_structure(p), idx)
    code = idx["code"].iloc[0]
    out = llm._prompt_candidate(  # pylint: disable=protected-access
        code, ["Example from search"]
    )
    assert code in out
    assert llm.soc[code].group_title in out
    assert "Example from search" in out


@pytest.mark.parametrize(
    "code, expected_output",
    [
        ("1", ["Code", "Title", "Details"]),
        ("1111", ["Code", "Title", "Details", "Includes"]),
    ],
)
@pytest.mark.llm
def test_prompt_candidate_include_all(prompt_candidate_soc, code, expected_output):
    """include_all adds Details and Includes (tasks) like SIC _prompt_candidate."""
    result = prompt_candidate_soc._prompt_candidate(  # pylint: disable=protected-access
        code, ["Example title"], include_all=True
    )
    assert isinstance(result, str)
    assert all(part in result for part in expected_output)


@pytest.mark.llm
def test_model_name():
    assert (
        ClassificationLLM(classification_type="soc").llm.model_name
        == "gemini-2.5-flash"
    )


# Tests for rising errors
@pytest.mark.llm
def test_open_api_key_raise_not_implemented_error():
    with pytest.raises(
        NotImplementedError,
        match="Need to provide an OpenAI API key",
    ):
        ClassificationLLM(classification_type="soc", model_name="gpt")


@pytest.mark.llm
def test_model_family_raise_not_implemented_error():
    with pytest.raises(
        NotImplementedError,
        match="Unsupported model family",
    ):
        ClassificationLLM(classification_type="soc", model_name="aaaa")


@pytest.fixture
async def classification_llm_with_soc_unambiguous(mocker, mock_soc):  # pylint: disable=W0621
    """ClassificationLLM with mocked ainvoke for unambiguous_soc_code."""
    mock_object_dict = {
        "codable": False,
        "class_code": None,
        "class_descriptive": None,
        "alt_candidates": [
            {
                "class_code": "1111",
                "class_descriptive": "description",
                "likelihood": 0.5,
            }
        ],
        "reasoning": "This is reasoning for the llm answer. Padded to 50 characters (Pydantic)",
    }
    mock_object_json = json.dumps(mock_object_dict)
    mock_message = mocker.Mock(spec=AIMessage)
    mock_message.content = mock_object_json
    mocker.patch(
        "survey_assist_classification_core.llm.llm.ChatVertexAI.ainvoke",
        return_value=mock_message,
    )
    llm_class = ClassificationLLM(classification_type="soc", model_name=MODEL_NAME)
    llm_class.soc = mock_soc
    return llm_class


@pytest.fixture
def prompt_candidate_soc(mock_soc):  # pylint: disable=W0621
    """LLM with SOC hierarchy attached."""
    llm_class = ClassificationLLM(classification_type="soc", model_name=MODEL_NAME)
    llm_class.soc = mock_soc
    return llm_class


@pytest.mark.llm
async def test_llm_response_mocked_unambiguous_soc_code(
    classification_llm_with_soc_unambiguous,
):
    """Mocked unambiguous_soc_code returns typed response and call dict."""
    result = await classification_llm_with_soc_unambiguous.unambiguous_soc_code(
        industry_descr="",
        semantic_search_results=[],
        job_description="",
        job_title="",
    )
    assert isinstance(result[0], UnambiguousResponse)
    assert isinstance(result[1], dict)


@pytest.mark.parametrize(
    "title, expected_job_title",
    [
        ("", "Unknown"),
        (" ", "Unknown"),
        (None, "Unknown"),
        ("teacher", "teacher"),
    ],
)
@pytest.mark.llm
async def test_unambiguous_soc_code_call_dict_job_title_correct(
    title,
    expected_job_title,
    classification_llm_with_soc_unambiguous,
):
    """job_title in call dict matches normalisation rules."""
    result = (
        await classification_llm_with_soc_unambiguous.unambiguous_soc_code(
            "school",
            [{"title": "Teaching", "code": "1111"}],
            title,
            "educate kids",
        )
    )[1]["job_title"]
    assert result == expected_job_title


@pytest.mark.llm
async def test_unambiguous_soc_code_followup_is_str(
    classification_llm_with_soc_unambiguous,
):
    """Reasoning on the unambiguous response is a string."""
    result = (
        await classification_llm_with_soc_unambiguous.unambiguous_soc_code(
            industry_descr="school",
            semantic_search_results=[{"title": "Teaching", "code": "1111"}],
            job_title="teacher",
            job_description="educate kids",
        )
    )[0].reasoning
    assert isinstance(result, str)


@pytest.mark.llm
async def test_llm_response_mocked_formulate_open_question(
    mocker, prompt_candidate_soc
):
    """formulate_open_question returns typed response and call dict with mocked output."""
    mock_object_dict = {"class_code": "", "class_descriptive": "", "likelihood": 0.5}
    mock_object_json = json.dumps(mock_object_dict)

    mock_message = mocker.Mock(spec=AIMessage)
    mock_message.content = mock_object_json

    mocker.patch(
        "survey_assist_classification_core.llm.llm.ChatVertexAI.ainvoke",
        return_value=mock_message,
    )

    result = await prompt_candidate_soc.formulate_open_question(
        industry_descr="",
        job_title="",
        job_description="",
        llm_output="",
    )
    assert isinstance(result[0], OpenFollowUp)
    assert isinstance(result[1], dict)


@pytest.fixture
def mock_soc():
    """Minimal SOC hierarchy from the packaged example lookup table."""
    ref = ("occupational_classification", "data/example_soc_lookup_data.csv")
    with as_file(files(ref[0]).joinpath(ref[1])) as path:
        p = str(path)
        idx = lib_load_soc_index(p)
        soc = load_hierarchy(lib_load_soc_structure(p), idx)
    return soc
