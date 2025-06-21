# ================================================
# Arquivo dataset-statistics.py
#
# Descrição:
# -  Análise Descritiva do Dataset com geração de gráficos em arquivos PNG
# 
# Uso:
# Execute com Python 3.10 ou superior
#
# Observações:
# - Os gráficos serão salvos na pasta ./graficos/
# ================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
import io

os.makedirs("graficos", exist_ok=True)

dataset = '2023/diabetes_binary_5050split_health_indicators_BRFSS2023.csv'
dados = pd.read_csv(dataset, sep=',')

def salvar_informacoes_gerais_png(df, variaveis_verificacao, arquivo_saida="graficos/informacoes_gerais.png"):
    info_buffer = []

    buffer = io.StringIO()
    df.info(buf=buffer)
    info_buffer.append("== Informações Gerais ==")
    info_buffer.extend(buffer.getvalue().splitlines())

    info_buffer.append("\n== Estatísticas Descritivas ==")
    describe_text = df.describe().T.round(2).to_string()
    info_buffer.extend(describe_text.splitlines())

    info_buffer.append("\n== Valores Únicos ==")
    for var in variaveis_verificacao:
        if var in df.columns:
            valores = ", ".join(map(str, sorted(df[var].dropna().unique())))
            texto = f"{var}: {valores}"
            info_buffer.extend(wrap(texto, width=120))
        else:
            info_buffer.append(f"{var}: [variável não encontrada]")

    fig, ax = plt.subplots(figsize=(12, 20))
    ax.axis("off")
    full_text = "\n".join(info_buffer)
    ax.text(0, 1, full_text, fontsize=10, va="top", family="monospace")

    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Informações gerais salvas em: {arquivo_saida}")

os.makedirs("graficos", exist_ok=True)
variaveis_verificacao = ['IMC', 'Saúde_Geral', 'Saúde_Mental', 'Saúde_Física', 'Idade', 'Nível_Educação', 'Renda']
salvar_informacoes_gerais_png(dados, variaveis_verificacao)

# Configurações de estilo
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Tipos de variáveis
tipos_variaveis = {
    "Diabetes_binário": "Binária",
    "Pressão_Alta": "Binária",
    "Colesterol_Alto": "Binária",
    "Avaliou_Colesterol": "Binária",
    "IMC": "Quantitativa contínua",
    "Fumante": "Binária",
    "Ataque_Cardíaco": "Binária",
    "Doença_Coronário_ouInfarto": "Binária",
    "Atividade_Física": "Binária",
    "Consumo_Álcool": "Binária",
    "Seguro_Saúde": "Binária",
    "Acesso_Saúde": "Binária",
    "Saúde_Geral": "Ordinal",
    "Saúde_Mental": "Quantitativa Discreta",
    "Saúde_Física": "Quantitativa Discreta",
    "Dificuldade_Andar": "Binária",
    "Gênero": "Binária",
    "Idade": "Ordinal",
    "Nível_Educação": "Ordinal",
    "Renda": "Ordinal"
}

# Estatísticas + gráficos
for var, tipo in tipos_variaveis.items():
    print(f"\n🔹 Variável: {var} ({tipo})")

    if var not in dados.columns:
        print("Variável não encontrada no dataset.")
        continue

    serie = dados[var].dropna()

    if tipo.startswith("Quantitativa"):
        print(serie.describe())
        print("Moda:", serie.mode().values[0] if not serie.mode().empty else "N/A")

        # Histograma
        plt.figure()
        sns.histplot(serie, kde=True, bins=20, color="steelblue")
        plt.title(f"Histograma: {var}")
        plt.xlabel(var)
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(f"graficos/{var}_histograma.png")
        plt.close()

    else:  # Binária ou Ordinal
        counts = serie.value_counts().sort_index()
        percents = counts / counts.sum() * 100

        print("Frequência (%):")
        print(percents.round(2))
        print("Moda:", serie.mode().values[0])

        # Gráfico de barras
        plt.figure()
        sns.countplot(x=serie, order=sorted(serie.unique()), palette="pastel")
        plt.title(f"Distribuição de {var}")
        plt.xlabel(var)
        plt.ylabel("Contagem")
        plt.tight_layout()
        plt.savefig(f"graficos/{var}_barras.png")
        plt.close()

        # Gráfico de pizza
        plt.figure()
        counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, colors=sns.color_palette("pastel"))
        plt.title(f"Distribuição de {var}")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(f"graficos/{var}_pizza.png")
        plt.close()
