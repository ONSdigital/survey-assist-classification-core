"""Shared prompt fragments for SIC and SOC classification LLM flows."""

from langchain_core.prompts import PromptTemplate

CORE_PROMPT = """You are a conscientious classification assistant of respondent data
for the use in the UK official statistics. Respondent data may be in English or Welsh,
but you always respond in British English."""

FIX_PARSING_PROMPT = PromptTemplate.from_template(
    """You are a meticulous assistant tasked with ensuring that
the output from a language model adheres strictly to the required JSON format.

Your task is to review the output and make any necessary adjustments to ensure it is valid JSON.
If the output is not valid JSON, you must fix it without altering the intended meaning.

====Output from LLM====
{llm_output}

===Output Format===
{format_instructions}
"""
)
