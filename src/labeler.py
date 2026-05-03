import re


def find_spans(text: str, inserted: dict[str, str]) -> list[dict]:
    """
    Finds character-level spans for each inserted PII value in text.
    Handles overlapping by keeping the longest match.
    Returns list of {"start": int, "end": int, "label": str}.
    """
    raw_spans = []
    for value, label in inserted.items():
        if not value:
            continue
        for m in re.finditer(re.escape(value), text):
            raw_spans.append({"start": m.start(), "end": m.end(), "label": label})

    raw_spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    result = []
    last_end = -1
    for span in raw_spans:
        if span["start"] >= last_end:
            result.append(span)
            last_end = span["end"]
    return sorted(result, key=lambda s: s["start"])


def to_entity_format(text: str, spans: list[dict]) -> dict:
    """Returns the NER JSONL format: {text, entities}."""
    return {"text": text, "entities": spans}


def label_text(text: str, inserted: dict[str, str]) -> dict:
    """Full pipeline: text + inserted PII dict → labeled example."""
    spans = find_spans(text, inserted)
    return to_entity_format(text, spans)
