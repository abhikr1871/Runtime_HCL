"""API contracts. FROZEN at 0:50 - change only with the whole team's agreement.

Field names in ChatRequest/ChatResponse follow the hackathon brief exactly;
extra response fields are additive so the judges' scripts keep working.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

AnswerType = Literal["recommendation", "comparison", "clarification", "product_fact", "not_found", "refusal"]
Language = Literal["en", "hi", "hinglish"]
Intent = Literal["search", "refine", "compare", "product_fact", "out_of_scope", "other_shopper_data", "greeting"]


# ---------- /chat ----------
class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    customer_id: str = Field(min_length=1, max_length=32)
    message: str = Field(min_length=1, max_length=1000)


class ProductOut(BaseModel):
    product_id: str
    name: str
    price: Optional[int] = None
    in_stock: bool
    reason: str
    brand: Optional[str] = None
    rating: Optional[float] = None
    category: Optional[str] = None


class Source(BaseModel):
    product_id: str
    fields: list[str]


class Funnel(BaseModel):
    total: int
    after_filters: int
    shown: int


class Relaxation(BaseModel):
    blocked_by: str
    message: str
    nearest: list[ProductOut] = []


class ComparisonTable(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]


class ChatResponse(BaseModel):
    # --- brief fields ---
    trace_id: str
    answer_type: AnswerType
    answer: str
    products: list[ProductOut] = []
    sources: list[Source] = []
    follow_up_question: Optional[str] = None
    # --- our additive fields ---
    language: Language = "en"
    active_filters: dict[str, Any] = {}
    funnel: Optional[Funnel] = None
    assumptions: list[str] = []
    relaxation: Optional[Relaxation] = None
    comparison: Optional[ComparisonTable] = None


# ---------- internal: understanding output ----------
class Slots(BaseModel):
    """What the shopper asked for. Produced by nlu_rules (+ nlu_llm fallback)."""
    intent: Intent = "search"
    language: Language = "en"
    query_text: str = ""
    category: Optional[str] = None
    product_type: Optional[str] = None
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    brands_include: list[str] = []
    brands_exclude: list[str] = []
    size: Optional[str] = None
    color: Optional[str] = None
    attributes: list[str] = []
    occasion: Optional[str] = None
    recipient: Optional[str] = None
    references: list[int] = []          # 0-based positions in the last result list
    product_ids: list[str] = []         # explicit IDs typed by the shopper, e.g. "P-0012"
    wants_cheaper: bool = False
    sort: Optional[Literal["price_asc", "price_desc", "rating_desc"]] = None
    confidence: float = 0.5
    source: Literal["rules", "llm", "rules+llm"] = "rules"


# ---------- internal: session ----------
class SessionState(BaseModel):
    session_id: str
    customer_id: str
    filters: dict[str, Any] = {}        # merged slots that act as filters
    last_results: list[str] = []        # ordered product IDs shown last turn
    asked_slots: list[str] = []
    clarify_streak: int = 0
    turn_count: int = 0


# ---------- catalogue ----------
class Product(BaseModel):
    product_id: str
    name: str
    category: str
    source: str = "ingest"
    subcategory: Optional[str] = None
    brand: Optional[str] = None
    price_inr: Optional[int] = None
    mrp_inr: Optional[int] = None
    stock: int = 0
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    description: str = ""
    description_lang: Language = "en"
    attributes: dict[str, Any] = {}
    sizes: list[str] = []
    colors: list[str] = []
    tags: list[str] = []
    weight_raw: Optional[str] = None
    weight_g: Optional[float] = None
    image_path: Optional[str] = None
    is_active: bool = True


class Review(BaseModel):
    review_id: str
    product_id: str
    rating: Optional[float] = None
    text: str
    lang: Language = "en"
    author_alias: Optional[str] = None
    injection_flag: bool = False


class ProductDetail(Product):
    reviews: list[Review] = []


# ---------- /ingest ----------
class IngestResult(BaseModel):
    created: int
    updated: int
    errors: list[dict[str, Any]] = []
    took_ms: int


# ---------- /health ----------
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    db: bool
    products: int
    text_index: str
    llm: str
    mock_llm: bool
