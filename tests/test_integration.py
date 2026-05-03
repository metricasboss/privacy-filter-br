"""
Integration test — calls real 4devs API and real Haiku API.
Run with: pytest tests/test_integration.py -v -m integration
Requires: ANTHROPIC_API_KEY in environment or .env file
"""
import pytest
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.integration


def test_generate_20_examples_real_apis():
    from src.generator import generate_example, GeneratorStats
    from src.haiku import HaikuGenerator

    haiku = HaikuGenerator()
    stats = GeneratorStats()
    examples = []
    attempts = 0

    while len(examples) < 20 and attempts < 60:
        attempts += 1
        ex = generate_example(haiku)
        stats.record(ex, "failed" if ex is None else None)
        if ex:
            examples.append(ex)

    assert len(examples) >= 15, f"Only got {len(examples)}/20 valid examples after {attempts} attempts"

    valid_labels = {
        "PRIVATE_CPF", "PRIVATE_CNPJ", "PRIVATE_RG", "PRIVATE_CNH",
        "PRIVATE_PIS", "PRIVATE_TITULO_ELEITOR", "PRIVATE_CERTIDAO",
        "PRIVATE_IE", "PRIVATE_PERSON", "PRIVATE_EMAIL",
        "PRIVATE_PHONE", "PRIVATE_ADDRESS", "PRIVATE_DATE"
    }

    for ex in examples:
        assert "text" in ex
        assert "entities" in ex
        assert len(ex["text"]) >= 50
        assert len(ex["entities"]) >= 2
        for ent in ex["entities"]:
            assert ent["label"] in valid_labels, f"Unknown label: {ent['label']}"

    acceptance_rate = len(examples) / attempts * 100
    print(f"\nAcceptance rate: {acceptance_rate:.0f}% ({len(examples)}/{attempts})")
    assert acceptance_rate >= 40, f"Acceptance rate too low: {acceptance_rate:.0f}%"

    Path("data").mkdir(exist_ok=True)
    with open("data/integration_sample.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Sample saved to data/integration_sample.jsonl")
