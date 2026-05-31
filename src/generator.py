from groq import Groq
from .config import GROQ_API_KEY, LLM_MODEL

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


#using multi line prompt + dynamic variable injection
def build_prompt(query, context):
    """Build strict prompt for grounded answers"""
    return f"""
You are a document QA assistant.

Rules:
1. Use ONLY the supplied context.
2. If the question asks for a specific number of items (e.g., "five things", "three rights"), 
    return EXACTLY that many.
3. If the context contains more items than requested, return only the first N items 
    (N = number in question).
4. If the context contains fewer items than requested, return all available items and note the 
    shortage.
5. Do NOT add extra text, repeat the question, or include unrelated content.
If information is absent, respond: NOT FOUND IN DOCUMENT


CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""


def stream_response(query, context):

    # Generate streaming response token by token
    prompt = build_prompt(query, context)
    
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        stream=True
    )
    
    #sending our response(token) as soon as they are generated
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content