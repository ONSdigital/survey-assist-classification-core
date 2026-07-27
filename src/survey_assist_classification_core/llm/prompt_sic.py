"""Module for generating prompt templates for SIC classification tasks.

This module provides prompt templates for Survey Assist SIC flows that remain in use:
unambiguous coding, SA RAG shortlisting, final assignment, and open follow-up.
"""

# pylint: disable=invalid-name,duplicate-code # Need to clean up the code to remove this

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from survey_assist_classification_core.llm.prompt_common import CORE_PROMPT
from survey_assist_classification_core.models.response_model import (
    FinalSICAssignment,
    OpenFollowUp,
    SicResponse,
    UnambiguousResponse,
)

_core_prompt = CORE_PROMPT

_sa_sic_template_rag = """"Given the respondent's description of the main activity their
company does, their job title and job description (which may be different to the
main company activity), your task is to determine a list of the most likely UK SIC
(Standard Industry Classification) codes for this company and the final code
that is most likely to match the description.

The following will be provided to make your decision and send appropriate output:
Respondent Data
Relevant subset of UK SIC 2007 (you must only use this list to classify)
Output Format (the output format MUST be valid JSON)

Only use the subset of UK SIC 2007 provided to determine if you can match the most
likely sic codes, provide a confidence score between 0 and 1 where 0.1 is least
likely and 0.9 is most likely.

You must return a subset list of possible sic codes (UK SIC 2007 codes provided)
that might match with a confidence score for each.

You must provide a follow up question that would help identify the exact coding based
on the list you respond with.

Always provide reasoning for your decision.


===Respondent Data===
- Company's main activity: {industry_descr}
- Job Title: {job_title}
- Job Description: {job_description}

===Relevant subset of UK SIC 2007===
{sic_index}

===Output Format===
{format_instructions}

===Output===
"""

parser = PydanticOutputParser(
    pydantic_object=SicResponse  # type: ignore # Suspect langchain ver bug
)

SA_SIC_PROMPT_RAG = PromptTemplate.from_template(
    template=_core_prompt + _sa_sic_template_rag,
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    },
)

_sic_template_unambiguous = """"You are an expert in industrial classifications.
You are tasked with determining whether a survey response can be assigned to a
single 5-digit UK Standard Industrial Classification (SIC) code based on initial respondent data alone.

Key objective:  Determine if the response can be coded unambiguously to a single 5-digit SIC code.

Assignment logic:
1. Code as unambiguous when response can be coded to a single 5-digit SIC code with 99
per cent confidence based on available evidence.
2. Code as uncodable to 5-digit when multiple candidates are plausible and
additional information is needed to distinguish between them.

===Analysis steps===
Follow these steps in order:
1. Review each candidate from the shortlist of relevant SIC codes against the respondent data.
2. Assess alignment - Consider:
   - Semantic similarity between respondent descriptions and SIC code descriptions
   - Job role compatibility with typical activities in each SIC code
   - Industry context alignment
   - Matches with specific examples listed under each code.
3. Assign confidence scores - Rate each candidate from 0.1 (least likely) to 0.9 (most likely).
4. Decide if response can be codeded unambiguously to a single 5-digit SIC code with 99 per cent confidence.
5. Provide reasoning for your decision.

===Respondent Data===
- Company's main activity: {industry_descr}
- Job Title: {job_title}
- Job Description: {job_description}

===Shortlist===
{sic_candidates}

===Output Format===
{format_instructions}
"""
parser_unambiguous = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
    pydantic_object=UnambiguousResponse
)

SIC_PROMPT_UNAMBIGUOUS = PromptTemplate.from_template(
    template=_core_prompt + _sic_template_unambiguous,
    partial_variables={
        "format_instructions": parser_unambiguous.get_format_instructions(),
    },
)

_sic_template_final_assignment = """"You are an expert in industrial classifications.
You are tasked with assigning UK Standard Industrial Classification (SIC) codes to survey
responses with high confidence.

Key objective: You MUST assign a 5-digit SIC code from the candidates provided. Only provide a higher-level
code if multiple candidates have nearly identical confidence scores (within 0.2 of each other) AND no single
can be identified as the clear best match.

Assignment logic:
1. Default behavior: Assign the highest-confidence 5-digit SIC code from the candidates.
2. Higher-level code exception: Only if two or more codes have confidence scores within 0.2
 of each other AND you cannot determine a clear winner. Provide the most granular
higher-level code with X padding to 5-digits (e.g., 8610X for 4-digit confidence, 86XXX for
3-digit confidence, 8XXXX for 2-digit confidence).
3. 95% confidence interpretation: This means "more likely than not" given the available evidence -
not absolute certainty.

Key principles:
1. Focus on Best Fit: Rather than seeking absolute certainty, identify which code best fits the totality of evidence.
2. Prioritise information on employer of the respondent rather than their specific role.
3. Be Decisive: The goal is accurate classification, not perfect certainty. If evidence clearly points to one
code over others, assign it confidently.

Important: When a respondent's closed question answer directly matches or closely aligns with a SIC code
description, this constitutes strong evidence for that code.

Follow these steps in order:
1. Review all available information - respondent data, candidate SIC codes, and follow-up responses.
2. Evaluate each candidate SIC code against all available evidence.
3. Assign confidence scores - Rate each candidate from 0.1 (least likely) to 0.9 (most likely).
Weight respondent's own descriptions heavily.
4. Apply assignment logic - Select the candidate with the highest confidence score as your primary assignment.
Only consider higher-level coding if multiple candidates have nearly identical scores (within 0.2) and you cannot
differentiate between them.
5. Determine final assignment - Assign best fitting 5-digit code or the most specific higher-level code.
6. Provide clear reasoning - Explain your decision with specific evidence.

===Respondent Data===
- Company's main activity: {industry_descr}
- Job Title: {job_title}
- Job Description: {job_description}

===Short list of UK SIC codes===
{sic_candidates}

===Follow up question 1===
{open_question}
{answer_to_open_question}

===Follow up question 2===
{closed_question}
{answer_to_closed_question}

===Output Format===
{format_instructions}
"""
parser_final_assignment = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
    pydantic_object=FinalSICAssignment
)

SIC_PROMPT_FINAL_ASSIGNMENT = PromptTemplate.from_template(
    template=_core_prompt + _sic_template_final_assignment,
    partial_variables={
        "format_instructions": parser_final_assignment.get_format_instructions(),
    },
)


_open_follow_up = """"You are an expert survey methodologist specialising in
    UK industrial classification (UK SIC). Generate one open-ended follow-up question
    to help assign the most relevant UK SIC code.

Objective
- Produce exactly one question that elicits the key information needed to distinguish
    between the shortlisted SIC candidates, focusing on the employer's main business activity.

Inputs
- Respondent data:
- Company's main activity: {industry_descr}
- Job title: {job_title}
- Job description: {job_description}
- Shortlist from previous model: {llm_output}
- Note: These are candidate SIC categories; do not mention codes or "SIC" to the respondent.

How to decide what to ask
- Identify the smallest, most informative difference among the candidates and target that with a single question.
- Prioritise discriminators in this order:
1) Stage in the value chain (e.g., manufacture/processing vs wholesale vs retail vs repair/installation vs
    rental/leasing vs publishing/software vs consultancy/training).
2) Main product or service category (what goods/services the employer mainly provides).
3) Main customer type (households vs businesses vs government/health/education).
4) Delivery mode or setting (on-site vs online; physical goods vs digital; own-brand vs third-party).
- Ask about only one discriminator—the one most likely to resolve the ambiguity.

Quality standards
- Language and clarity:
    - Use plain British English; avoid or define jargon and abbreviations.
    - Keep the single question concise (max 25 words), grammatically correct, and neutral.
    - Use "employer" for for-profit; use "organisation" for non-profits, charities, public bodies, and education.
        Default to "employer", if ambiguous.
    - Refer to the present situation (e.g., "currently", "main").
    - Do not mention SIC or any code numbers.
    - Do not ask for company names, client names, or other personal/sensitive data.
- Question structure:
    - Start with "What", "How", "Which", or "Where".
    - Focus on the employer's main business activities, products, or services—not the respondent's personal tasks.
    - One issue per question; no A/B or either/or phrasing; avoid binary questions.
    - Limit to one sentence ending with a question mark.
    - You may add one additional sentence with broad, non-leading examples covering a wide range of options;
        omit examples if they would be leading.
- Respondent considerations:
    - Make it easy to answer in a few words.
    - Ask only what a typical employee would reasonably know.
    - Avoid requiring calculations or percentages.

Edge cases
- If the shortlist is empty or clearly points to one category, ask a general clarifying question about
    the main product/service or value-chain stage to confirm classification.
- Do not output explanations or reasoning; only the formatted result.

Output format
- Return output that strictly follows:
{format_instructions}
"""
parser_followup_open = PydanticOutputParser(pydantic_object=OpenFollowUp)

SIC_PROMPT_OPENFOLLOWUP = PromptTemplate.from_template(
    template=_core_prompt + _open_follow_up,
    partial_variables={
        "format_instructions": parser_followup_open.get_format_instructions(),
    },
)


# pylint: disable=too-few-public-methods
class PromptTemplates:
    """A collection of predefined prompt templates for Survey Assist SIC flows.

    Attributes:
        SA_SIC_PROMPT_RAG (PromptTemplate): A prompt template for SA SIC with RAG.
        SIC_PROMPT_UNAMBIGUOUS (PromptTemplate): A prompt template for unambiguous
                                                    SIC classification.
        SIC_PROMPT_FINAL_ASSIGNMENT (PromptTemplate): Final SIC assignment prompt.
        SIC_PROMPT_OPENFOLLOWUP (PromptTemplate): Open follow-up question prompt.
    """

    def __init__(self):
        self.SA_SIC_PROMPT_RAG = SA_SIC_PROMPT_RAG
        self.SIC_PROMPT_UNAMBIGUOUS = SIC_PROMPT_UNAMBIGUOUS
        self.SIC_PROMPT_FINAL_ASSIGNMENT = SIC_PROMPT_FINAL_ASSIGNMENT
        self.SIC_PROMPT_OPENFOLLOWUP = SIC_PROMPT_OPENFOLLOWUP

    def get_all_templates(self) -> list[PromptTemplate]:
        """Returns all stored prompt templates as a list."""
        return [
            self.SA_SIC_PROMPT_RAG,
            self.SIC_PROMPT_UNAMBIGUOUS,
            self.SIC_PROMPT_FINAL_ASSIGNMENT,
            self.SIC_PROMPT_OPENFOLLOWUP,
        ]
