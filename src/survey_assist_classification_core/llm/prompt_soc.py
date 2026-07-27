"""Module for generating prompt templates for SOC classification tasks.

This module provides prompt templates for Survey Assist SOC flows that remain in use:
unambiguous coding and open follow-up.
"""

# pylint: disable=invalid-name,duplicate-code # Need to clean up the code to remove this

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from survey_assist_classification_core.llm.prompt_common import CORE_PROMPT
from survey_assist_classification_core.models.response_model import (
    OpenFollowUp,
    UnambiguousResponse,
)

_core_prompt = CORE_PROMPT

_soc_template_unambiguous = """"You are an expert in occupational classifications.
You are tasked with determining whether a survey response can be assigned to a
single four-digit UK Standard Occupational Classification (SOC 2020) unit group
based on initial respondent data alone.

Key objective: Determine if the response can be coded unambiguously to a single
four-digit SOC code.

Assignment logic:
1. Code as unambiguous when the response can be coded to a single four-digit SOC
code with 99 per cent confidence based on available evidence.
2. Code as uncodable when multiple candidates are plausible and additional
information is needed to distinguish between them.

===Analysis steps===
Follow these steps in order:
1. Review each candidate from the shortlist of relevant SOC codes against the respondent data.
2. Assess alignment - job role, tasks, industry context, and example activities.
3. Assign confidence scores from 0.1 (least likely) to 0.9 (most likely).
4. Decide if a single four-digit SOC code can be assigned with 99 per cent confidence.
5. Provide reasoning for your decision.

===Respondent Data===
- Company's main activity: {industry_descr}
- Job Title: {job_title}
- Job Description: {job_description}

===Shortlist===
{soc_candidates}

===Output Format===
{format_instructions}
"""
parser_unambiguous = PydanticOutputParser(
    pydantic_object=UnambiguousResponse  # type: ignore[arg-type]
)

SOC_PROMPT_UNAMBIGUOUS = PromptTemplate.from_template(
    template=_core_prompt + _soc_template_unambiguous,
    partial_variables={
        "format_instructions": parser_unambiguous.get_format_instructions(),
    },
)

_open_follow_up = """"You are an expert survey methodologist specialising in
UK occupational classification (SOC 2020). Generate one open-ended follow-up question
to help assign the most relevant four-digit SOC unit group.

Objective
- Produce exactly one question that elicits the key information needed to distinguish
between the shortlisted SOC candidates, focusing on the respondent's job and the
employer's context where relevant.

Inputs
- Respondent data:
- Company's main activity: {industry_descr}
- Job title: {job_title}
- Job description: {job_description}
- Shortlist from previous model: {llm_output}
- Note: These are candidate occupational categories; do not mention codes or "SOC"
to the respondent.

How to decide what to ask
- Identify the smallest, most informative difference among the candidates and target that with a single question.
- Prioritise discriminators in this order:
1) Main tasks and duties (what the respondent actually does day to day, including tools, materials, and who or what they work with).
2) Work setting or sector (for example healthcare, education, construction, agriculture, office, retail premises, outdoor or site-based work).
3) Skill level or seniority (for example supervisory or line-management vs hands-on; professional or technical vs elementary).
4) Who or what they mainly serve or produce for (patients, pupils, customers, machinery, data or systems, goods).
- Ask about only one discriminator—the one most likely to resolve the ambiguity.

Quality standards
- Language and clarity:
    - Use plain British English; avoid or define jargon and abbreviations.
    - Keep the single question concise (max 25 words), grammatically correct, and neutral.
    - Use "employer" for for-profit; use "organisation" for non-profits, charities, public bodies, and education.
        Default to "employer", if ambiguous.
    - Refer to the present situation (e.g., "currently", "main").
    - Do not mention SOC or any code numbers.
    - Do not ask for personal names, client names, or other sensitive data beyond what is needed to describe the job.
- Question structure:
    - Start with "What", "How", "Which", or "Where".
    - Focus on the respondent's job and main activities; use employer context only where it helps distinguish the role.
    - One issue per question; no A/B or either/or phrasing; avoid binary questions.
    - Limit to one sentence ending with a question mark.
    - You may add one additional sentence with broad, non-leading examples covering a wide range of options;
        omit examples if they would be leading.
- Respondent considerations:
    - Make it easy to answer in a few words.
    - Ask only what a typical employee would reasonably know about their own role.
    - Avoid requiring calculations or percentages.

Edge cases
- If the shortlist is empty or clearly points to one category, ask a general clarifying question about
    main tasks or work setting to confirm classification.
- Do not output explanations or reasoning; only the formatted result.

Output format
- Return output that strictly follows:
{format_instructions}
"""
parser_followup_open = PydanticOutputParser(pydantic_object=OpenFollowUp)

SOC_PROMPT_OPENFOLLOWUP = PromptTemplate.from_template(
    template=_core_prompt + _open_follow_up,
    partial_variables={
        "format_instructions": parser_followup_open.get_format_instructions(),
    },
)
