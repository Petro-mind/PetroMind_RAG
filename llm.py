from openai import OpenAI
from config import HF_BASE_URL, HF_API_KEY, LLM_MODEL, LLM_TEMP, LLM_MAX_TOKENS


def get_llm_client() -> OpenAI:
    """Create HuggingFace router client. Same as CairoS1."""
    return OpenAI(base_url=HF_BASE_URL, api_key=HF_API_KEY)


def topk_to_string(topk: list) -> str:
    """
    Format top-K chunks into context string for LLM.
    Mirror of CairoS1 topk_to_string().
    """
    parts = []
    for item in topk:
        block = (
            f"Unit ID      : {item['unit_id']}\n"
            f"Cycle        : {item['cycle_id']}\n"
            f"RUL          : {item['rul']:.1f} cycles remaining\n"
            f"Zone         : {item['zone']}\n"
            f"Failure mode : {item['failure_mode']}\n"
            f"Score        : {item['score']}\n"
            f"Content      :\n{item['chunk_text']}"
        )
        parts.append(block)
    return "\n\n---\n\n".join(parts)


def generate_text(
    context: str,
    question: str,
    client: OpenAI = None,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMP,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    """
    Generate grounded answer from retrieved sensor context.
    Mirror of CairoS1 generate_text().
    """
    if client is None:
        client = get_llm_client()

    prompt = f"""
You are PetroMind, an expert maintenance engineering AI assistant.
Answer questions using ONLY the sensor log records in the CONTEXT below.
These are real readings from turbofan engine degradation tests.

STRICT RULES:
- Use ONLY information present in the CONTEXT.
- Do NOT use external knowledge or guess.
- If the answer is not in the CONTEXT, say exactly:
  "The answer is not found in the provided sensor records."
- Always cite Unit ID, Cycle ID, and RUL for every fact you state.
- Be concise and actionable for engineers in the field.

OUTPUT FORMAT:
Answer:
<your answer — use bullet points for multiple facts>

Source:
Unit ID      : <unit_id>
Cycle        : <cycle_id>
RUL          : <rul>
Failure mode : <failure_mode>

--------------------------------

CONTEXT:
{context}

QUESTION:
{question}
""".strip()

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"
