import json
import re

from groq import Groq

from .config import GROQ_API_KEY, LLM_MODEL

client = Groq(api_key=GROQ_API_KEY)


def evaluate_answer(question, context, answer, ground_truth):

    prompt = f"""
You are an expert evaluator for Retrieval-Augmented Generation (RAG).

Evaluate the generated answer.

QUESTION
--------
{question}

GROUND TRUTH
------------
{ground_truth}

RETRIEVED CONTEXT
-----------------
{context}

GENERATED ANSWER
----------------
{answer}

Evaluate the answer using these criteria.

1. Groundedness
- Is the answer supported by the retrieved context?

2. Relevance
- Does it answer the user's question?

3. Completeness
- Does it include all important information?

4. Correctness
- Is it consistent with the ground truth?

Give each criterion a score from 1 to 10.

Return ONLY a JSON object in this format:

{{
    "groundedness": 9,
    "relevance": 10,
    "completeness": 8,
    "correctness": 9,
    "overall": 9,
    "feedback": "Short explanation."
}}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    # print("\n========== RAW RESPONSE ==========\n")
    # print(content)
    # print("\n==================================\n")

    # Remove markdown code fences if present
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)

    # Extract JSON object
    match = re.search(r"\{.*\}", content, re.DOTALL)

    if not match:
        raise ValueError(
            f"Groq did not return valid JSON.\n\nReturned:\n{content}"
        )

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON returned by Groq.\n\nReturned:\n{match.group()}"
        ) from e
#If no errors occur, the function returns the evaluation results as a **Python dictionary**.
