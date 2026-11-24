"""LangGraph sales-assistant agent for Modern Furniture Pacific.

Model: AWS Bedrock Claude Haiku 4.5 (us-east-1) via LangChain's
ChatBedrockConverse, wrapped in LangGraph's ReAct agent.
"""
from __future__ import annotations

import os
import re
from typing import Any

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from app.tools import (
    TOOLS,
    build_whatsapp_shortlist_link,
    get_seen_products,
    reset_seen_products,
    set_budget_kes,
)

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
# NOTE: the bare model ID (anthropic.claude-haiku-4-5-20251001-v1:0) is NOT
# invocable via Converse on-demand — Bedrock requires the cross-region
# inference profile ID (us. prefix). Verified live 2026-11-25.
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

SYSTEM_PROMPT = """You are the Modern Furniture Pacific sales assistant, a friendly \
sales assistant for Modern Furniture Pacific in Nairobi, Kenya. Tagline: \
"Make you feel luxury". We sell premium sofas, beds, dining sets and tables.

Rules you must follow:
- Answer ONLY from tool output. Prices come from the product tools, never \
from memory. If a tool did not return a price, say you do not have it and \
offer to check on WhatsApp.
- NEVER pitch a sold-out product. If the customer asks for a product that is \
out of stock, say in your first sentence that it is out of stock, then \
IMMEDIATELY call search_products for \
in-stock alternatives in the same category and name the best one in the \
same reply with its price. Do not just ask whether they want alternatives.
- Delivery fees, discounts, negotiation and stock quantities are NOT in your \
data. Never invent them. Hand those questions off to WhatsApp \
(0716 335555) and let the team answer.
- Prefer premium studio images over watermarked catalogue photos when \
describing products (the premium image and 3D view are shown with each product card).
- When presenting a product, also pitch 1 or 2 related in-stock pieces (calling \
search_products in the same or a complementary category, respecting any stated budget).
- Once the customer states a budget, remember it and filter ALL subsequent searches \
and recommendations to stay within that budget (pass max_price to search_products).
- When the customer is ready to buy or asks to order, hand off to WhatsApp \
(0716 335555) with a prefilled shortlist link: https://wa.me/254716335555?text=... \
listing their chosen in-stock items and prices.
- Be concise and friendly. Prices in Kenyan shillings (KSh). Keep answers \
short: name the product, its price or price range, and one line on why it \
fits, then point to WhatsApp to order.
- Use the tools to look products up. When a customer names a product, call \
get_product. When they describe what they want, call search_products. \
Compare with compare_products when they ask to choose between products.
"""


_BUDGET_RE = re.compile(
    r"(?:budget(?:\s+is|\s+of)?|under|below|max(?:imum)?|up\s*to|within|around|spend)\s*"
    r"(?:ksh\.?|kes)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k)?\b",
    re.IGNORECASE,
)


def extract_budget_kes(messages: list[dict[str, str]]) -> float | None:
    """Return the most recently stated customer budget in KES, if any."""
    budget: float | None = None
    for m in messages:
        if m.get("role", "user") != "user":
            continue
        for match in _BUDGET_RE.finditer(m.get("content", "")):
            raw = float(match.group(1).replace(",", ""))
            if match.group(2):
                raw *= 1000
            if raw > 0:
                budget = raw
    return budget


def _build_agent():
    llm = ChatBedrockConverse(
        model=MODEL_ID,
        region_name=AWS_REGION,
        temperature=0.3,
        max_tokens=2048,
    )
    return create_react_agent(llm, TOOLS)


_AGENT = None


def get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = _build_agent()
    return _AGENT


def run_chat(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Run one chat turn. Returns {"answer": str, "products": [cards], "whatsapp_shortlist_link": str | None}."""
    reset_seen_products()
    budget = extract_budget_kes(messages)
    set_budget_kes(budget)
    agent = get_agent()
    system_prompt = SYSTEM_PROMPT
    if budget:
        system_prompt += (
            f"\nActive customer budget: KSh {int(budget):,}. "
            f"Filter all searches and recommendations to <= KSh {int(budget):,}."
        )
    lc_messages: list = [SystemMessage(content=system_prompt)]
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    # Max ~5 tool iterations: each iteration is a few graph steps.
    result = agent.invoke(
        {"messages": lc_messages}, config={"recursion_limit": 25}
    )
    answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            if isinstance(msg.content, str):
                answer = msg.content
            elif isinstance(msg, AIMessage) and isinstance(msg.content, list):
                answer = "".join(
                    b.get("text", "") for b in msg.content if isinstance(b, dict)
                )
            break
    products = get_seen_products()
    return {
        "answer": answer,
        "products": products,
        "whatsapp_shortlist_link": build_whatsapp_shortlist_link(products),
    }
