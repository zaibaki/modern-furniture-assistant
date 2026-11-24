"""LangGraph sales-assistant agent for Modern Furniture Pacific.

Model: AWS Bedrock Claude Haiku 4.5 (us-east-1) via LangChain's
ChatBedrockConverse, wrapped in LangGraph's ReAct agent.
"""
from __future__ import annotations

import os
from typing import Any

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from app.tools import TOOLS, get_seen_products, reset_seen_products

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
describing products (the premium image is shown with each product card).
- Be concise and friendly. Prices in Kenyan shillings (KSh). Keep answers \
short: name the product, its price or price range, and one line on why it \
fits, then point to WhatsApp to order.
- Use the tools to look products up. When a customer names a product, call \
get_product. When they describe what they want, call search_products. \
Compare with compare_products when they ask to choose between products.
"""


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
    """Run one chat turn. Returns {"answer": str, "products": [cards]}."""
    reset_seen_products()
    agent = get_agent()
    lc_messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
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
            elif isinstance(msg.content, list):
                answer = "".join(
                    b.get("text", "") for b in msg.content if isinstance(b, dict)
                )
            break
    return {"answer": answer, "products": get_seen_products()}
