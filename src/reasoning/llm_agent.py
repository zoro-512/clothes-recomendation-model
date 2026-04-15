"""
LLM Reasoning Pipeline using LangChain.

Generates natural language explanations for recommendations.
Falls back gracefully if no LLM API key is configured.
"""

import os
from typing import Optional


def build_explanation_chain():
    """
    Build a LangChain pipeline to generate item recommendation explanations.
    Requires OPENAI_API_KEY env var. Falls back to a rule-based explanation if missing.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=120,
        )

        prompt = ChatPromptTemplate.from_template(
            "Customer Request: {query}\n"
            "Recommended Item: {item_name}\n"
            "Item Details: {item_desc}\n"
            "Customer's Favourite Season: {season}\n\n"
            "You are a personal AI fashion stylist. In exactly one warm, engaging sentence, "
            "explain why this specific item is perfect for the customer's request."
        )

        chain = prompt | llm
        return chain, "openai"

    except Exception:
        return None, "fallback"


def explain_recommendation(
    query: str,
    item_name: str,
    item_desc: str,
    season: str = "unknown",
    chain=None,
    mode: str = "fallback",
) -> str:
    """
    Generate a natural language explanation for why an item is being recommended.

    Args:
        query: The user's original search or question.
        item_name: Product name.
        item_desc: Product description.
        season: User's favourite season from their profile.
        chain: LangChain chain object (from build_explanation_chain).
        mode: 'openai' or 'fallback'.

    Returns:
        A single-sentence explanation string.
    """
    if mode == "openai" and chain is not None:
        try:
            result = chain.invoke(
                {
                    "query": query,
                    "item_name": item_name,
                    "item_desc": item_desc,
                    "season": season,
                }
            )
            return result.content.strip()
        except Exception as e:
            print(f"[LLM] OpenAI call failed: {e}. Using fallback.")

    # Rule-based fallback (no API key needed)
    return (
        f"We recommend '{item_name}' because it closely matches your request for '{query}' "
        f"and is a popular choice among customers with similar preferences."
    )


def parse_natural_query(query: str, chain=None, mode: str = "fallback") -> dict:
    """
    Use the LLM to extract structured intent from a natural language query.
    E.g., 'warm jacket for Iceland trip' → {intent: 'outerwear', season: 'winter'}
    """
    if mode == "openai" and chain is not None:
        try:
            from langchain.prompts import ChatPromptTemplate

            parse_prompt = ChatPromptTemplate.from_template(
                "User query: {query}\n\n"
                "Extract structured intent. Reply ONLY in this format:\n"
                "intent: <clothing_type>\nseason: <winter/summer/spring/autumn/unknown>"
            )
            parse_chain = parse_prompt | chain.steps[0]  # Reuse the LLM
            result = parse_chain.invoke({"query": query})
            lines = result.content.strip().splitlines()
            parsed = {}
            for line in lines:
                if ":" in line:
                    k, v = line.split(":", 1)
                    parsed[k.strip()] = v.strip()
            return parsed
        except Exception:
            pass
    return {"intent": query, "season": "unknown"}
