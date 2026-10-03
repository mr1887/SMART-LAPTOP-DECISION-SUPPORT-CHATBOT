from typing import Literal, Optional, Union
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

AllowedField = Literal[
    "price",
    "ram_gb",
    "storage_gb",
    "weight_kg",
    "battery_minutes",
    "gpu_discrete",
    "gpu_keyword",
]

ALLOWED_FIELDS = {
    "price",
    "ram_gb",
    "storage_gb",
    "weight_kg",
    "battery_minutes",
    "gpu_discrete",
    "gpu_keyword",
}


class Constraint(BaseModel):
    field: AllowedField
    operator: Literal["<=", ">=", "="]
    value: Union[int, float, str, bool]
    type: Literal["hard", "soft"] = "hard"
    source_text: Optional[str] = None


class Preference(BaseModel):
    field: AllowedField
    direction: Literal["minimize", "maximize", "prefer"]
    source_text: Optional[str] = None


class RequirementSet(BaseModel):
    constraints: list[Constraint] = Field(default_factory=list)
    preferences: list[Preference] = Field(default_factory=list)
    required_tags: list[str] = Field(default_factory=list)
