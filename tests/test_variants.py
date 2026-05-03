from src.variants import variantes_cpf, variantes_cnpj, variantes_rg, variantes_cep, variantes_telefone, pick_variant


def test_variantes_cpf_retorna_quatro_formatos():
    vs = variantes_cpf("680.075.670-97")
    assert len(vs) == 4
    assert ("680.075.670-97", "PRIVATE_CPF") in vs
    assert ("68007567097", "PRIVATE_CPF") in vs
    assert ("680.075.***-**", "PRIVATE_CPF") in vs
    assert ("680 075 670 97", "PRIVATE_CPF") in vs


def test_variantes_cnpj_retorna_quatro_formatos():
    vs = variantes_cnpj("72.682.864/0001-41")
    assert len(vs) == 4
    assert ("72.682.864/0001-41", "PRIVATE_CNPJ") in vs
    assert ("72682864000141", "PRIVATE_CNPJ") in vs
    assert ("72.682.864/****-**", "PRIVATE_CNPJ") in vs
    assert ("72 682 864 0001 41", "PRIVATE_CNPJ") in vs


def test_variantes_rg():
    vs = variantes_rg("27.141.489-3")
    assert len(vs) == 4
    assert ("27.141.489-3", "PRIVATE_RG") in vs
    assert ("271414893", "PRIVATE_RG") in vs


def test_variantes_cep():
    vs = variantes_cep("01310-100")
    assert len(vs) == 3
    assert ("01310-100", "PRIVATE_ADDRESS") in vs
    assert ("01310100", "PRIVATE_ADDRESS") in vs


def test_variantes_telefone():
    vs = variantes_telefone("(11) 99999-9999")
    assert len(vs) == 4
    assert ("(11) 99999-9999", "PRIVATE_PHONE") in vs
    assert ("11999999999", "PRIVATE_PHONE") in vs


def test_pick_variant_retorna_um_dos_valores():
    vs = variantes_cpf("680.075.670-97")
    valor, label = pick_variant(vs)
    assert label == "PRIVATE_CPF"
    assert valor in [v for v, _ in vs]
