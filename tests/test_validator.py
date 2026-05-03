from src.validator import validate_example, ValidationResult


def test_valid_example_passes():
    example = {
        "text": "Contrato com João Silva, CPF 123.456.789-09, residente à Av Paulista, 1000",
        "entities": [
            {"start": 13, "end": 23, "label": "PRIVATE_PERSON"},
            {"start": 29, "end": 43, "label": "PRIVATE_CPF"}
        ]
    }
    result = validate_example(example)
    assert result == ValidationResult.VALID


def test_span_out_of_bounds_fails():
    example = {
        "text": "CPF 123 and some additional text to make this long enough for validation checks",
        "entities": [{"start": 4, "end": 150, "label": "PRIVATE_CPF"}]
    }
    result = validate_example(example)
    assert result == ValidationResult.SPAN_OUT_OF_BOUNDS


def test_empty_text_fails():
    example = {"text": "", "entities": []}
    result = validate_example(example)
    assert result == ValidationResult.TEXT_TOO_SHORT


def test_short_text_fails():
    example = {"text": "Oi", "entities": []}
    result = validate_example(example)
    assert result == ValidationResult.TEXT_TOO_SHORT


def test_no_entities_fails():
    example = {"text": "Um texto longo sem nenhum dado pessoal identificado aqui neste documento", "entities": []}
    result = validate_example(example)
    assert result == ValidationResult.NO_ENTITIES


def test_overlapping_spans_fail():
    example = {
        "text": "João Silva mora aqui nesta cidade de São Paulo mesmo",
        "entities": [
            {"start": 0, "end": 10, "label": "PRIVATE_PERSON"},
            {"start": 5, "end": 10, "label": "PRIVATE_PERSON"}
        ]
    }
    result = validate_example(example)
    assert result == ValidationResult.OVERLAPPING_SPANS
