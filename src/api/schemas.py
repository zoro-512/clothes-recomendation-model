"""
Pydantic schemas for the FastAPI recommendation API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class RecommendationRequest(BaseModel):
    query: str = Field(..., example="I need a warm waterproof jacket for Iceland")
    user_id: Optional[str] = Field(None, example="USER_001")
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations to return")
    season: Optional[str] = Field(None, example="winter")
    gender: Optional[str] = Field(None, example="men")
    min_price: Optional[float] = Field(None, example=500.0, description="Minimum price filter")
    max_price: Optional[float] = Field(None, example=2000.0, description="Maximum price filter")
    brand: Optional[str] = Field(None, example="Nike", description="Single brand filter (legacy)")
    brands: Optional[List[str]] = Field(None, example=["Nike", "Adidas"], description="Multiple brand filter")


class RecommendedItem(BaseModel):
    article_id: str
    product_name: str
    product_group: Optional[str]
    product_type: Optional[str]
    gender_category: Optional[str]
    description: Optional[str]
    price: Optional[float] = None
    original_price: Optional[float] = None
    rank_score: float
    semantic_score: float
    product_url: Optional[str] = None
    explanation: str


class RecommendationResponse(BaseModel):
    query: str
    user_id: Optional[str]
    recommendations: List[RecommendedItem]
    is_fallback: bool = False
    total_candidates_evaluated: int


class FeedbackRequest(BaseModel):
    user_id: str = Field(..., example="USER_001")
    article_id: str = Field(..., example="0718673002")
    action: str = Field(..., example="purchase", description="One of: click, cart, purchase")


class FeedbackResponse(BaseModel):
    article_id: str
    action: str
    reward_points: int
    message: str
