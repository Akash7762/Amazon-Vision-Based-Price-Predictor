"""Response shapes. FastAPI uses them for validation and for the /docs page."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PriceRange(BaseModel):
    low: float = Field(description="Low end of the range, USD")
    high: float = Field(description="High end of the range, USD")
    coverage: float = Field(
        description="Share of validation products with a similar estimate whose real price "
                    "fell inside such a range (0.8 = 8 in 10)")


class Prediction(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "price": 12.48, "currency": "USD",
        "range": {"low": 5.24, "high": 34.32, "coverage": 0.8},
        "model_version": "v0.1-model"}})

    price: float = Field(description="Estimated price, USD")
    currency: Literal["USD"] = "USD"
    range: PriceRange
    model_version: str


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str
    model_sha256: str
    input_size: int
    range_coverage: float


class ErrorResponse(BaseModel):
    detail: str
