"""Validation helpers for child-owned tool input schemas on AgentCards."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolInputSchemaValidation:
    """Validation result for a ``tool_input_schema`` payload."""

    normalized: dict[str, Any] | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def validate_tool_input_schema(value: Any) -> ToolInputSchemaValidation:
    """Validate an optional ``tool_input_schema`` payload.

    Returns a normalized schema when valid, plus non-fatal warning messages for
    weak-but-usable schema metadata.
    """

    if value is None:
        return ToolInputSchemaValidation(normalized=None, errors=(), warnings=())

    if not isinstance(value, Mapping):
        return ToolInputSchemaValidation(
            normalized=None,
            errors=("must be a mapping",),
            warnings=(),
        )

    schema = dict(value)
    errors: list[str] = []
    warnings: list[str] = []

    schema_type = schema.get("type")
    if schema_type != "object":
        errors.append("'type' must be 'object'")

    properties_value = schema.get("properties")
    properties: dict[str, Any] = {}
    if properties_value is None:
        properties = {}
    elif isinstance(properties_value, Mapping):
        properties = dict(properties_value)
    else:
        errors.append("'properties' must be a mapping when present")

    required_value = schema.get("required")
    required_names: list[str] = []
    if required_value is not None:
        if not isinstance(required_value, list):
            errors.append("'required' must be a list of property names")
        else:
            for index, entry in enumerate(required_value):
                if not isinstance(entry, str) or not entry.strip():
                    errors.append(f"'required[{index}]' must be a non-empty string")
                else:
                    required_names.append(entry)

    required_set = set(required_names)
    unknown_required = sorted(name for name in required_set if name not in properties)
    if unknown_required:
        errors.append(
            "'required' references undefined properties: "
            + ", ".join(unknown_required)
        )

    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_name, str) or not prop_name.strip():
            errors.append("'properties' keys must be non-empty strings")
            continue

        if not isinstance(prop_schema, Mapping):
            errors.append(f"'properties.{prop_name}' must be a mapping")
            continue

        if prop_name in required_set:
            description = prop_schema.get("description")
            if not isinstance(description, str) or not description.strip():
                warnings.append(
                    f"required property '{prop_name}' should include a description"
                )

    additional_properties = schema.get("additionalProperties")
    if additional_properties is not None and not isinstance(
        additional_properties, (bool, Mapping)
    ):
        errors.append("'additionalProperties' must be a boolean or mapping")

    normalized = schema if not errors else None
    return ToolInputSchemaValidation(
        normalized=normalized,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )

