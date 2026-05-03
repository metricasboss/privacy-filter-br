import pytest
from unittest.mock import patch, Mock
from src.fodevs import Fodevs4


def test_gerar_cpf_returns_formatted_string():
    with patch("src.fodevs.requests.post") as mock_post:
        mock_post.return_value.text = "680.075.670-97"
        mock_post.return_value.raise_for_status = Mock()
        f = Fodevs4()
        result = f.gerar_cpf()
    assert result == "680.075.670-97"
    assert len(result.replace(".", "").replace("-", "")) == 11


def test_gerar_cnpj_returns_formatted_string():
    with patch("src.fodevs.requests.post") as mock_post:
        mock_post.return_value.text = "72.682.864/0001-41"
        mock_post.return_value.raise_for_status = Mock()
        f = Fodevs4()
        result = f.gerar_cnpj()
    assert result == "72.682.864/0001-41"


def test_gerar_pessoa_returns_dict_with_required_fields():
    with patch("src.fodevs.requests.post") as mock_post:
        mock_post.return_value.text = '[{"nome":"João Silva","cpf":"123.456.789-09","rg":"12.345.678-9","email":"joao@email.com","celular":"(11) 99999-9999","telefone_fixo":"(11) 3333-3333","cep":"01310-100","endereco":"Av Paulista","numero":1000,"bairro":"Bela Vista","cidade":"São Paulo","estado":"SP","data_nasc":"01/01/1990","pis":"123.45678.90-1","cnh":"12345678901"}]'
        mock_post.return_value.raise_for_status = Mock()
        f = Fodevs4()
        result = f.gerar_pessoa(estado="SP")
    assert result["nome"] == "João Silva"
    assert result["cpf"] == "123.456.789-09"
    assert result["estado"] == "SP"
    for field in ["nome", "cpf", "rg", "email", "celular", "cep", "endereco", "cidade", "estado", "data_nasc", "pis"]:
        assert field in result, f"Missing field: {field}"


def test_gerar_rg_returns_string():
    with patch("src.fodevs.requests.post") as mock_post:
        mock_post.return_value.text = "27.141.489-3"
        mock_post.return_value.raise_for_status = Mock()
        f = Fodevs4()
        result = f.gerar_rg()
    assert isinstance(result, str)
    assert len(result) > 5


def test_gerar_ie_returns_string_for_sp():
    with patch("src.fodevs.requests.post") as mock_post:
        mock_post.return_value.text = "818.800.070.960"
        mock_post.return_value.raise_for_status = Mock()
        f = Fodevs4()
        result = f.gerar_ie(estado="SP")
    assert result == "818.800.070.960"


def test_gerar_certidao_returns_string():
    with patch("src.fodevs.requests.post") as mock_post:
        mock_post.return_value.text = "256757 01 55 2022 1 08962 159 2623902-70"
        mock_post.return_value.raise_for_status = Mock()
        f = Fodevs4()
        result = f.gerar_certidao(tipo="nascimento")
    assert "2022" in result


from src.pessoa import gerar_perfil_completo


def test_gerar_perfil_completo_has_all_pii():
    with patch("src.pessoa.Fodevs4") as MockFodevs:
        inst = MockFodevs.return_value
        inst.gerar_pessoa.return_value = {
            "nome": "Maria Silva", "cpf": "123.456.789-09",
            "rg": "12.345.678-9", "email": "maria@email.com",
            "celular": "(11) 99999-9999", "telefone_fixo": "(11) 3333-3333",
            "cep": "01310-100", "endereco": "Av Paulista", "numero": 1000,
            "bairro": "Bela Vista", "cidade": "São Paulo", "estado": "SP",
            "data_nasc": "01/01/1990", "pis": "123.45678.90-1", "cnh": "12345678901"
        }
        inst.gerar_ie.return_value = "818.800.070.960"
        inst.gerar_titulo_eleitor.return_value = "228214550167"
        inst.gerar_certidao.return_value = "256757 01 55 2022 1 08962 159 2623902-70"
        inst.gerar_cnpj.return_value = "72.682.864/0001-41"

        perfil = gerar_perfil_completo(estado="SP")

    assert perfil["nome"] == "Maria Silva"
    assert perfil["cpf"] == "123.456.789-09"
    assert perfil["ie"] == "818.800.070.960"
    assert perfil["titulo_eleitor"] == "228214550167"
    assert perfil["certidao_nascimento"] == "256757 01 55 2022 1 08962 159 2623902-70"
    assert perfil["cnpj"] == "72.682.864/0001-41"
    assert "estado" in perfil
