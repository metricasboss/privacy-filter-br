import re
import random


def variantes_cpf(cpf: str) -> list[tuple[str, str]]:
    raw = re.sub(r"[.\-]", "", cpf)
    parcial = cpf[:8] + "***-**"
    espacos = cpf.replace(".", " ").replace("-", " ")
    return [(cpf, "PRIVATE_CPF"), (raw, "PRIVATE_CPF"),
            (parcial, "PRIVATE_CPF"), (espacos, "PRIVATE_CPF")]


def variantes_cnpj(cnpj: str) -> list[tuple[str, str]]:
    raw = re.sub(r"[.\-/]", "", cnpj)
    parcial = cnpj[:11] + "****-**"
    espacos = re.sub(r"[.\-/]", " ", cnpj)
    return [(cnpj, "PRIVATE_CNPJ"), (raw, "PRIVATE_CNPJ"),
            (parcial, "PRIVATE_CNPJ"), (espacos, "PRIVATE_CNPJ")]


def variantes_rg(rg: str) -> list[tuple[str, str]]:
    raw = re.sub(r"[.\-]", "", rg)
    parcial = rg[:5] + "***-*"
    espacos = rg.replace(".", " ").replace("-", " ")
    return [(rg, "PRIVATE_RG"), (raw, "PRIVATE_RG"),
            (parcial, "PRIVATE_RG"), (espacos, "PRIVATE_RG")]


def variantes_cep(cep: str) -> list[tuple[str, str]]:
    raw = cep.replace("-", "")
    espacos = cep.replace("-", " ")
    return [(cep, "PRIVATE_ADDRESS"), (raw, "PRIVATE_ADDRESS"),
            (espacos, "PRIVATE_ADDRESS")]


def variantes_telefone(tel: str) -> list[tuple[str, str]]:
    raw = re.sub(r"[\s()\-]", "", tel)
    sem_ddd = re.sub(r"^\(\d{2}\)\s*", "", tel)
    espacos = re.sub(r"[\-()\s]+", " ", tel).strip()
    return [(tel, "PRIVATE_PHONE"), (raw, "PRIVATE_PHONE"),
            (sem_ddd, "PRIVATE_PHONE"), (espacos, "PRIVATE_PHONE")]


def variantes_pis(pis: str) -> list[tuple[str, str]]:
    raw = re.sub(r"[.\-]", "", pis)
    return [(pis, "PRIVATE_PIS"), (raw, "PRIVATE_PIS")]


def variantes_cnh(cnh: str) -> list[tuple[str, str]]:
    return [(cnh, "PRIVATE_CNH"), (cnh[:5] + "******", "PRIVATE_CNH")]


def variantes_titulo(titulo: str) -> list[tuple[str, str]]:
    return [(titulo, "PRIVATE_TITULO_ELEITOR"),
            (titulo[:4] + "****" + titulo[-4:], "PRIVATE_TITULO_ELEITOR")]


def variantes_certidao(cert: str) -> list[tuple[str, str]]:
    return [(cert, "PRIVATE_CERTIDAO"),
            (re.sub(r"\s", "", cert), "PRIVATE_CERTIDAO")]


def variantes_ie(ie: str) -> list[tuple[str, str]]:
    raw = re.sub(r"[.\-/]", "", ie)
    return [(ie, "PRIVATE_IE"), (raw, "PRIVATE_IE")]


def pick_variant(variants: list[tuple[str, str]]) -> tuple[str, str]:
    return random.choice(variants)


def get_variants_for_perfil(perfil: dict) -> dict[str, list[tuple[str, str]]]:
    """Returns all format variants for every PII field in a perfil."""
    return {
        "cpf": variantes_cpf(perfil["cpf"]),
        "cnpj": variantes_cnpj(perfil["cnpj"]),
        "rg": variantes_rg(perfil["rg"]),
        "cep": variantes_cep(perfil["cep"]),
        "celular": variantes_telefone(perfil["celular"]),
        "telefone_fixo": variantes_telefone(perfil["telefone_fixo"]) if perfil.get("telefone_fixo") else [],
        "pis": variantes_pis(perfil["pis"]),
        "cnh": variantes_cnh(perfil["cnh"]),
        "titulo_eleitor": variantes_titulo(perfil["titulo_eleitor"]),
        "certidao_nascimento": variantes_certidao(perfil["certidao_nascimento"]),
        "ie": variantes_ie(perfil["ie"]),
    }
