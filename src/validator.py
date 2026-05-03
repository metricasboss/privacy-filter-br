from enum import Enum

MIN_TEXT_LENGTH = 50


class ValidationResult(Enum):
    VALID = "valid"
    TEXT_TOO_SHORT = "text_too_short"
    NO_ENTITIES = "no_entities"
    SPAN_OUT_OF_BOUNDS = "span_out_of_bounds"
    OVERLAPPING_SPANS = "overlapping_spans"


def validate_example(example: dict) -> ValidationResult:
    text = example.get("text", "")
    entities = example.get("entities", [])

    if len(text) < MIN_TEXT_LENGTH:
        return ValidationResult.TEXT_TOO_SHORT

    if not entities:
        return ValidationResult.NO_ENTITIES

    sorted_ents = sorted(entities, key=lambda e: e["start"])
    last_end = -1
    for ent in sorted_ents:
        if ent["end"] > len(text):
            return ValidationResult.SPAN_OUT_OF_BOUNDS
        if ent["start"] < last_end:
            return ValidationResult.OVERLAPPING_SPANS
        last_end = ent["end"]

    return ValidationResult.VALID
