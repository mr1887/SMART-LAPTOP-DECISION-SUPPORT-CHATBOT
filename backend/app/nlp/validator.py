from typing import Any, Union
from app.nlp.schema import ALLOWED_FIELDS, Constraint, Preference, RequirementSet

VALID_OPERATORS = {"<=", ">=", "="}
VALID_TYPES = {"hard", "soft"}
VALID_DIRECTIONS = {"minimize", "maximize", "prefer"}


def _to_number(val: Any) -> Union[int, float, None]:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return val
    if isinstance(val, str):
        val_str = val.strip().replace(",", "")
        try:
            if "." in val_str:
                return float(val_str)
            return int(val_str)
        except ValueError:
            return None
    return None


def _validate_field_value(field: str, val: Any) -> Union[int, float, str, bool, None]:
    if field in ("price", "ram_gb", "storage_gb", "battery_minutes"):
        num = _to_number(val)
        if num is not None and num > 0:
            return num
        return None

    if field == "weight_kg":
        num = _to_number(val)
        if num is not None and 0.5 <= num <= 10.0:
            return float(num) if isinstance(num, float) else num
        return None

    if field == "gpu_discrete":
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            lower = val.strip().lower()
            if lower in ("true", "1", "yes", "co", "có"):
                return True
            if lower in ("false", "0", "no", "khong", "không"):
                return False
        if isinstance(val, (int, float)):
            return bool(val)
        return None

    if field == "gpu_keyword":
        if isinstance(val, str) and val.strip():
            return val.strip()
        return None

    return None


def validate_requirement_set(
    data: Union[RequirementSet, dict[str, Any]]
) -> RequirementSet:
    """
    Chuẩn hóa và lọc các ràng buộc, sở thích không hợp lệ:
    - Loại bỏ field, operator, type không nằm trong định nghĩa.
    - Ép kiểu số và kiểm tra miền giá trị hợp lệ.
    - Trả về đối tượng RequirementSet hợp lệ và deterministic.
    """
    if isinstance(data, RequirementSet):
        raw_dict = data.model_dump()
    elif isinstance(data, dict):
        raw_dict = data
    else:
        return RequirementSet()

    validated_constraints: list[Constraint] = []
    for c in raw_dict.get("constraints", []):
        if isinstance(c, Constraint):
            c_dict = c.model_dump()
        elif isinstance(c, dict):
            c_dict = c
        else:
            continue

        field = c_dict.get("field")
        op = c_dict.get("operator")
        raw_val = c_dict.get("value")
        c_type = c_dict.get("type", "hard")
        source_text = c_dict.get("source_text")

        if field not in ALLOWED_FIELDS:
            continue
        if op not in VALID_OPERATORS:
            continue
        if c_type not in VALID_TYPES:
            c_type = "hard"

        val = _validate_field_value(field, raw_val)
        if val is None:
            continue

        validated_constraints.append(
            Constraint(
                field=field,
                operator=op,
                value=val,
                type=c_type,
                source_text=str(source_text) if source_text is not None else None,
            )
        )

    validated_preferences: list[Preference] = []
    for p in raw_dict.get("preferences", []):
        if isinstance(p, Preference):
            p_dict = p.model_dump()
        elif isinstance(p, dict):
            p_dict = p
        else:
            continue

        field = p_dict.get("field")
        direction = p_dict.get("direction")
        source_text = p_dict.get("source_text")

        if field not in ALLOWED_FIELDS:
            continue
        if direction not in VALID_DIRECTIONS:
            continue

        validated_preferences.append(
            Preference(
                field=field,
                direction=direction,
                source_text=str(source_text) if source_text is not None else None,
            )
        )

    raw_tags = raw_dict.get("required_tags", [])
    validated_tags = [
        str(tag).strip()
        for tag in raw_tags
        if isinstance(tag, (str, int)) and str(tag).strip()
    ]

    return RequirementSet(
        constraints=validated_constraints,
        preferences=validated_preferences,
        required_tags=validated_tags,
    )
