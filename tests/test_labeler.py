from src.labeler import find_spans, to_entity_format, label_text


def test_find_spans_single_match():
    text = "CPF 680.075.670-97 do cliente"
    spans = find_spans(text, {"680.075.670-97": "PRIVATE_CPF"})
    assert spans == [{"start": 4, "end": 18, "label": "PRIVATE_CPF"}]


def test_find_spans_multiple_matches():
    text = "João Silva tem CPF 123.456.789-09 e CNPJ 72.682.864/0001-41"
    spans = find_spans(text, {
        "João Silva": "PRIVATE_PERSON",
        "123.456.789-09": "PRIVATE_CPF",
        "72.682.864/0001-41": "PRIVATE_CNPJ"
    })
    assert len(spans) == 3
    labels = {s["label"] for s in spans}
    assert labels == {"PRIVATE_PERSON", "PRIVATE_CPF", "PRIVATE_CNPJ"}


def test_find_spans_missing_value_returns_empty():
    text = "Texto sem PII"
    spans = find_spans(text, {"680.075.670-97": "PRIVATE_CPF"})
    assert spans == []


def test_find_spans_overlapping_uses_longest():
    text = "João Silva mora aqui"
    spans = find_spans(text, {
        "João Silva": "PRIVATE_PERSON",
        "João": "PRIVATE_PERSON"
    })
    assert len(spans) == 1
    assert spans[0]["end"] - spans[0]["start"] == len("João Silva")


def test_to_entity_format_output_structure():
    text = "CPF 680.075.670-97"
    spans = [{"start": 4, "end": 18, "label": "PRIVATE_CPF"}]
    result = to_entity_format(text, spans)
    assert result["text"] == text
    assert result["entities"] == spans


def test_label_text_full_pipeline():
    text = "Contrato com Maria Silva, CPF 123.456.789-09"
    inserted = {"Maria Silva": "PRIVATE_PERSON", "123.456.789-09": "PRIVATE_CPF"}
    result = label_text(text, inserted)
    assert result["text"] == text
    assert len(result["entities"]) == 2
