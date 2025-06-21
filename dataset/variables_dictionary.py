# ================================================
# Arquivo variables-dictionary.py
#
# Descrição:
# -  Representação das variáveis presentes no Dataset
# 
# Uso:
# Execute com Python 3.10 ou superior
#
# Observações:
# - (...)
# ================================================

# Tipos de variáveis

variaveis_binarias = {
    "Diabetes_binário": {
        "tipo": "Binária",
        "valores": {
            0: "Não possui Diabetes ou apenas Pré-Diabetes",
            1: "Possui Diabetes"
        }
    },
    "Pressão_Alta": {
        "tipo": "Binária",
        "valores": {
            0: "Não possui Pressão Alta",
            1: "Possui Pressão Alta"
        }
    },
    "Colesterol_Alto": {
        "tipo": "Binária",
        "valores": {
            0: "Não possui Colesterol Alto",
            1: "Possui Colesterol Alto"
        }
    },
    "Avaliou_Colesterol": {
        "tipo": "Binária",
        "valores": {
            0: "Não avaliou o Colesterol nos últimos 5 anos",
            1: "Avaliou o Colesterol nos últimos 5 anos"
        }
    },
    "Fumante": {
        "tipo": "Binária",
        "valores": {
            0: "Nunca fumou pelo menos 100 cigarros na vida",
            1: "Já fumou pelo menos 100 cigarros na vida"
        }
    },
    "Ataque_Cardíaco": {
        "tipo": "Binária",
        "valores": {
            0: "Já teve um ataque cardíaco",
            1: "Nunca teve um ataque cardíaco"
        }
    },
    "Doença_Coronário_ouInfarto": {
        "tipo": "Binária",
        "valores": {
            0: "Nunca teve CHD ou MI",
            1: "Já teve CHD ou MI"
        }
    },
    "Atividade_Física": {
        "tipo": "Binária",
        "valores": {
            0: "Realizou atividade física nos últimos 30 dias",
            1: "Não realizou atividade física nos últimos 30 dias"
        }
    },
    "Consumo_Álcool": {
        "tipo": "Binária",
        "valores": {
            0: "Alto consumo de álcool por semana",
            1: "Baixo consumo de álcool por semana"
        }
    },
    "Seguro_Saúde": {
        "tipo": "Binária",
        "valores": {
            0: "Tem acesso a Seguro Saúde",
            1: "Não tem acesso a Seguro Saúde"
        }
    },
    "Acesso_Saúde": {
        "tipo": "Binária",
        "valores": {
            0: "Tem acesso à Médicos e Hospitais",
            1: "Não tem acesso à Médicos e Hospitais por motivos financeiros"
        }
    },
    "Dificuldade_Andar": {
        "tipo": "Binária",
        "valores": {
            0: "Apresenta dificuldade para andar ou subir escadas",
            1: "Não apresenta dificuldade"
        }
    },
    "Gênero": {
        "tipo": "Binária",
        "valores": {
            0: "Feminino",
            1: "Masculino"
        }
    }
}

variaveis_ordinais = {
    "Saúde_Geral": {
        "tipo": "Ordinal",
        "valores": {
            1: "Excelente Saúde",
            2: "Muito boa",
            3: "Boa",
            4: "Justa",
            5: "Ruim"
        }
    },
    "Idade": {
        "tipo": "Ordinal",
        "valores": {
            1: "18 a 24 anos",
            2: "25 a 29 anos",
            3: "30 a 34 anos",
            4: "35 a 39 anos",
            5: "40 a 44 anos",
            6: "45 a 49 anos",
            7: "50 a 54 anos",
            8: "55 a 59 anos",
            9: "60 a 64 anos",
            10: "65 a 69 anos",
            11: "70 a 74 anos",
            12: "75 a 79 anos",
            13: "80 anos ou mais"
        }
    },
    "Nível_Educação": {
        "tipo": "Ordinal",
        "valores": {
            1: "Nunca frequentou escola ou apenas jardim de infância",
            2: "Entre a 1º e 8º série",
            3: "Entre 9º ao 2º ano do Ensino Médio",
            4: "Ensino Médio completo",
            5: "Ensino Superior incompleto ou Técnico (1 a 3 anos)",
            6: "Ensino Superior completo (4 anos ou mais)"
        }
    },
    "Renda": {
        "tipo": "Ordinal",
        "valores": {
            1: "Menos que 10.000 USD",
            2: "Entre 10.000 a 15.000 USD",
            3: "Entre 15.000 a 20.000 USD",
            4: "Entre 20.000 a 25.000 USD",
            5: "Entre 25.000 a 35.000 USD",
            6: "Entre 35.000 a 50.000 USD",
            7: "Entre 50.000 a 75.000 USD",
            8: "Entre 75.000 a 100.000 USD",
            9: "Entre 100.000 a 150.000 USD",
            10: "Entre 150.000 a 200.000 USD",
            11: "Mais que 200.000 USD"
        }
    }
}

variaveis_quantitativas = {
    "IMC": {
        "tipo": "Quantitativa contínua",
        "faixa": "1 - 9999 (valores de BMI multiplicados por 100)"
    },
    "Saúde_Mental": {
        "tipo": "Quantitativa discreta",
        "faixa": "0 - 30 dias com saúde mental ruim nos últimos 30 dias"
    },
    "Saúde_Física": {
        "tipo": "Quantitativa discreta",
        "faixa": "0 - 30 dias com saúde física ruim nos últimos 30 dias"
    }
}
