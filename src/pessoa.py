import random
from src.fodevs import Fodevs4

ESTADOS_BR = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO", "DF", "AM",
               "CE", "PE", "PA", "MT", "MS", "ES", "PB", "RN", "AL", "PI",
               "TO", "SE", "RO", "AC", "AP", "RR", "MA"]


def gerar_perfil_completo(estado: str | None = None) -> dict:
    """Generates a complete synthetic BR person profile with all PII types."""
    fodevs = Fodevs4()
    estado = estado or random.choice(ESTADOS_BR)

    pessoa = fodevs.gerar_pessoa(estado=estado)

    perfil = {
        "nome": pessoa["nome"],
        "cpf": pessoa["cpf"],
        "rg": pessoa["rg"],
        "email": pessoa["email"],
        "celular": pessoa["celular"],
        "telefone_fixo": pessoa.get("telefone_fixo", ""),
        "cep": pessoa["cep"],
        "endereco": f"{pessoa['endereco']}, {pessoa['numero']}, {pessoa['bairro']}",
        "cidade": pessoa["cidade"],
        "estado": estado,
        "data_nasc": pessoa.get("data_nasc", ""),
        "pis": pessoa.get("pis", fodevs.gerar_pis()),
        "cnh": pessoa.get("cnh", fodevs.gerar_cnh()),
        "ie": fodevs.gerar_ie(estado=estado),
        "titulo_eleitor": fodevs.gerar_titulo_eleitor(estado=estado),
        "certidao_nascimento": fodevs.gerar_certidao(tipo="nascimento"),
        "cnpj": fodevs.gerar_cnpj(),
    }
    return perfil
