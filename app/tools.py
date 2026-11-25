"""Supabase PostgREST tools for the Modern Furniture Pacific sales assistant.

All five tools are read-only. Product rows come from the live Supabase
backend (project hpdcfkhwznlcecjtdtid); prices are never invented — every
price the assistant quotes comes from one of these calls.
"""
from __future__ import annotations

import contextvars
import json
import os
import urllib.parse
from typing import Any

import httpx
from langchain_core.tools import tool

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
BASE = f"{SUPABASE_URL}/rest/v1" if SUPABASE_URL else ""

WHATSAPP_NUMBER = "254716335555"

# Products the tools returned during the current /chat run. Reset at the
# start of each request (see app/agent.py); main.py reads it to build the
# response's `products` array.
_seen_products: contextvars.ContextVar[list[dict[str, Any]]] = contextvars.ContextVar(
    "mfp_seen_products", default=[]
)
_budget_kes: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "mfp_budget_kes", default=None
)


def reset_seen_products() -> None:
    _seen_products.set([])


def get_seen_products() -> list[dict[str, Any]]:
    return list(_seen_products.get())


def set_budget_kes(budget: float | None) -> None:
    _budget_kes.set(budget)


def get_budget_kes() -> float | None:
    return _budget_kes.get()


def build_whatsapp_shortlist_link(products: list[dict[str, Any]]) -> str | None:
    """Build a prefilled WhatsApp link for the customer's in-stock shortlist."""
    shortlist = [p for p in products if p.get("in_stock", True)]
    if not shortlist:
        return None
    lines = ["Hi, I would like to order from Modern Furniture Pacific:"]
    for p in shortlist[:5]:
        lines.append(f"- {p['title']} ({p['price']})")
    text = urllib.parse.quote("\n".join(lines))
    return f"https://wa.me/{WHATSAPP_NUMBER}?text={text}"


def _headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Accept": "application/json",
    }


def _get(path: str, params: dict[str, Any] | None = None, service_role: bool = False) -> Any:
    key = SUPABASE_SERVICE_ROLE_KEY if service_role else SUPABASE_ANON_KEY
    if not BASE or not key:
        raise RuntimeError("SUPABASE_URL / Supabase key are not configured")
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(f"{BASE}{path}", headers=headers, params=params or {})
        resp.raise_for_status()
        return resp.json()


PRODUCT_SELECT = (
    "handle,title,original_title,type,price_min,price_max,in_stock,"
    "is_new,is_best_seller,description,"
    "categories(name,slug),"
    "product_variants(option,price,available,position)"
)


def _category_name(row: dict[str, Any]) -> str:
    cat = row.get("categories")
    if isinstance(cat, dict):
        return cat.get("name") or ""
    if isinstance(cat, list) and cat:
        return cat[0].get("name") or ""
    return ""


def _fmt_price(value: float | int) -> str:
    return f"KSh {int(value):,}"


def _price_label(row: dict[str, Any]) -> str:
    lo, hi = row["price_min"], row["price_max"]
    if lo == hi:
        return _fmt_price(lo)
    return f"{_fmt_price(lo)} - {_fmt_price(hi)}"


def _card(row: dict[str, Any], include_variants: bool = False) -> dict[str, Any]:
    handle = row["handle"]
    title = row["title"]
    price = _price_label(row)
    text = urllib.parse.quote(f"Hi, I would like to order the {title} ({price})")
    card: dict[str, Any] = {
        "handle": handle,
        "title": title,
        "category": _category_name(row),
        "price_min": row["price_min"],
        "price_max": row["price_max"],
        "price": price,
        "in_stock": row["in_stock"],
        "is_best_seller": row.get("is_best_seller", False),
        "is_new": row.get("is_new", False),
        "premium_image": f"/images/premium/{handle}-premium.jpg",
        "image_3d": f"/images/premium/{handle}-3d.jpg",
        "product_url": f"/products/{handle}",
        "whatsapp_link": f"https://wa.me/{WHATSAPP_NUMBER}?text={text}",
    }
    if include_variants:
        variants = sorted(row.get("product_variants") or [], key=lambda v: v.get("position", 0))
        card["variants"] = [
            {
                "option": v["option"],
                "price": v["price"],
                "available": v.get("available", True),
            }
            for v in variants
        ]
        if row.get("description"):
            card["description"] = row["description"]
    return card


def _record(rows: list[dict[str, Any]], include_variants: bool = False) -> None:
    seen = _seen_products.get()
    have = {p["handle"] for p in seen}
    for row in rows:
        if row["handle"] not in have:
            seen.append(_card(row, include_variants=include_variants))
            have.add(row["handle"])
    _seen_products.set(seen)


def _fetch_all_products() -> list[dict[str, Any]]:
    rows = _get(
        "/products",
        {
            "select": PRODUCT_SELECT,
            "product_variants.order": "position.asc",
            "order": "price_min.asc",
        },
    )
    return rows if isinstance(rows, list) else []


@tool
def search_products(
    query: str = "",
    category: str = "",
    max_price: float = 0,
    min_price: float = 0,
    in_stock_only: bool = True,
    limit: int = 5,
) -> str:
    """Search Modern Furniture Pacific products by keyword, category and price.

    Args:
        query: keywords to match against product titles (e.g. "7 seater sofa", "bed").
        category: category name filter, e.g. "Living Room", "Bedroom", "Dining", "Tables".
        max_price: only products whose lowest price is at most this (KES). 0 = no limit.
        min_price: only products whose highest price is at least this (KES). 0 = no limit.
        in_stock_only: when true (default), exclude sold-out products.
        limit: maximum number of products to return (default 5).
    Returns a JSON list of matching products with handle, title, category,
    price_min, price_max, in_stock, is_best_seller and is_new.
    """
    rows = _fetch_all_products()
    active_budget = get_budget_kes()
    effective_max = max_price
    if active_budget and active_budget > 0:
        effective_max = min(max_price, active_budget) if max_price > 0 else active_budget
    words = [w for w in query.lower().split() if w]
    out = []
    for row in rows:
        hay = f"{row['title']} {row['handle']} {row.get('type') or ''}".lower()
        if words and not all(w in hay for w in words):
            continue
        if category and category.lower() not in _category_name(row).lower():
            continue
        if effective_max and row["price_min"] > effective_max:
            continue
        if min_price and row["price_max"] < min_price:
            continue
        if in_stock_only and not row["in_stock"]:
            continue
        out.append(row)
    out = out[: max(1, limit)]
    _record(out)
    return json.dumps(
        [
            {
                "handle": r["handle"],
                "title": r["title"],
                "category": _category_name(r),
                "price_min": r["price_min"],
                "price_max": r["price_max"],
                "in_stock": r["in_stock"],
                "is_best_seller": r.get("is_best_seller", False),
                "is_new": r.get("is_new", False),
            }
            for r in out
        ]
    )


@tool
def get_product(handle: str) -> str:
    """Get full details for one product by its handle (e.g. "nova-sofa-set").

    Returns the product with all size/option variants and their prices,
    stock status, and description. Use this when the customer asks about a
    specific product by name.
    """
    rows = _get(
        "/products",
        {
            "select": PRODUCT_SELECT,
            "handle": f"eq.{handle}",
            "product_variants.order": "position.asc",
        },
    )
    if not rows:
        # Forgiving fallback: the model often guesses a handle. Match the
        # handle's words against product titles instead of failing outright.
        words = [w for w in handle.lower().replace("-", " ").split() if w]
        candidates = []
        for row in _fetch_all_products():
            hay = f"{row['title']} {row['handle']}".lower()
            if words and all(w in hay for w in words):
                candidates.append(row)
        if len(candidates) == 1:
            rows = candidates
        elif candidates:
            return json.dumps(
                {
                    "error": f"No product with handle '{handle}'. Close matches:",
                    "matches": [
                        {"handle": r["handle"], "title": r["title"]} for r in candidates
                    ],
                }
            )
        else:
            return json.dumps({"error": f"No product found with handle '{handle}'"})
    _record(rows, include_variants=True)
    row = rows[0]
    return json.dumps(
        {
            "handle": row["handle"],
            "title": row["title"],
            "category": _category_name(row),
            "price_min": row["price_min"],
            "price_max": row["price_max"],
            "in_stock": row["in_stock"],
            "is_best_seller": row.get("is_best_seller", False),
            "is_new": row.get("is_new", False),
            "description": row.get("description"),
            "variants": [
                {
                    "option": v["option"],
                    "price": v["price"],
                    "available": v.get("available", True),
                }
                for v in sorted(
                    row.get("product_variants") or [], key=lambda v: v.get("position", 0)
                )
            ],
        }
    )


@tool
def compare_products(handles: list[str]) -> str:
    """Compare 2-3 products side by side by their handles.

    Returns each product's title, category, price range, stock status and
    variants so the customer can see the differences.
    """
    handles = list(handles)[:3]
    rows: list[dict[str, Any]] = []
    for handle in handles:
        found = _get(
            "/products",
            {
                "select": PRODUCT_SELECT,
                "handle": f"eq.{handle}",
                "product_variants.order": "position.asc",
            },
        )
        rows.extend(found or [])
    _record(rows, include_variants=True)
    return json.dumps(
        [
            {
                "handle": r["handle"],
                "title": r["title"],
                "category": _category_name(r),
                "price_min": r["price_min"],
                "price_max": r["price_max"],
                "in_stock": r["in_stock"],
                "variants": [
                    {"option": v["option"], "price": v["price"]}
                    for v in sorted(
                        r.get("product_variants") or [], key=lambda v: v.get("position", 0)
                    )
                ],
            }
            for r in rows
        ]
    )


@tool
def list_categories() -> str:
    """List the product categories available at Modern Furniture Pacific."""
    rows = _get("/categories", {"select": "name,slug", "order": "name.asc"})
    return json.dumps([{"name": r["name"], "slug": r["slug"]} for r in rows])


@tool
def get_business_info() -> str:
    """Get the store's business info: name, tagline, phones, WhatsApp number,
    email, opening hours and showroom locations."""
    rows = _get("/business_info", {"select": "*", "limit": 1})
    if not rows:
        return json.dumps({"error": "No business info found"})
    row = rows[0]
    return json.dumps(
        {
            "name": row.get("name"),
            "tagline": row.get("tagline"),
            "phone": row.get("phone"),
            "whatsapp": row.get("whatsapp"),
            "email": row.get("email"),
            "hours": row.get("hours"),
            "showrooms": row.get("showrooms"),
        }
    )


def _norm_phone_tail(phone: str) -> str:
    digits = "".join(ch for ch in phone if ch.isdigit())
    return digits[-9:] if len(digits) >= 9 else digits


@tool
def lookup_order(reference: str, phone: str) -> str:
    """Look up a customer's submitted order status after verifying both order
    reference (e.g. 'MFP-AB12CD') and phone number.
    """
    ref = reference.strip().upper()
    tail = _norm_phone_tail(phone)
    if not ref or len(tail) < 7:
        return json.dumps(
            {
                "verified": False,
                "error": "Both a valid order reference (e.g. MFP-XXXXXX) and phone number are required.",
            }
        )
    if not SUPABASE_SERVICE_ROLE_KEY:
        return json.dumps(
            {
                "verified": False,
                "error": "Order lookup is temporarily unavailable. Please escalate to a human on WhatsApp (0716 335555).",
            }
        )
    rows = _get(
        "/orders",
        {"select": "*", "reference": f"eq.{ref}", "limit": 1},
        service_role=True,
    )
    if not rows:
        return json.dumps(
            {
                "verified": False,
                "error": f"No order found with reference '{ref}'. Please double-check the reference.",
            }
        )
    row = rows[0]
    if _norm_phone_tail(row.get("phone", "")) != tail:
        return json.dumps(
            {
                "verified": False,
                "error": "The phone number does not match this order reference.",
            }
        )
    return json.dumps(
        {
            "verified": True,
            "reference": row["reference"],
            "status": row["status"],
            "product_title": row["product_title"],
            "product_price": row.get("product_price"),
            "option": row.get("option"),
            "quantity": row.get("quantity", 1),
            "customer_name": row["customer_name"],
            "location": row["location"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "whatsapp_link": row.get("whatsapp_link"),
        }
    )


@tool
def escalate_to_human(
    reason: str,
    customer_name: str = "",
    phone: str = "",
    order_reference: str = "",
) -> str:
    """Escalate the conversation to a human sales representative on WhatsApp.

    ONLY call this when the customer explicitly asks to speak with a human,
    or when the issue cannot be resolved with the available tools. Never
    auto-invoke this for normal product or order-status questions.
    """
    lines = ["Hi Modern Furniture Pacific team, I need human assistance:"]
    if order_reference.strip():
        lines.append(f"- Order Ref: {order_reference.strip().upper()}")
    if customer_name.strip():
        lines.append(f"- Name: {customer_name.strip()}")
    if phone.strip():
        lines.append(f"- Phone: {phone.strip()}")
    lines.append(f"- Details: {reason.strip() or 'Customer requested human support'}")
    wa_link = f"https://wa.me/{WHATSAPP_NUMBER}?text={urllib.parse.quote(chr(10).join(lines))}"
    return json.dumps(
        {
            "escalated": True,
            "whatsapp_link": wa_link,
            "phone": "0716 335555",
            "email": "info@modernfurniturepacificltd.com",
            "hours": "Mon-Sat 9am-10pm, Sun 10am-6pm",
        }
    )


TOOLS = [
    search_products,
    get_product,
    compare_products,
    list_categories,
    get_business_info,
    lookup_order,
    escalate_to_human,
]
