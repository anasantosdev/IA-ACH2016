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
# - Usa dicionário de variáveis importado do arquivo variables-dicitionary.py
# ================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from textwrap import wrap
import io

# Importa os dicionários das variáveis
from variables_dictionary import variaveis_binarias, variaveis_ordinais, variaveis_quantitativas

# Junta todos os tipos de variáveis em um único dicionário
variaveis_dict = {}

for var, props in variaveis_binarias.items():
    variaveis_dict[var] = props["tipo"]
for var, props in variaveis_ordinais.items():
    variaveis_dict[var] = props["tipo"]
for var, props in variaveis_quantitativas.items():
    variaveis_dict[var] = props["tipo"]

os.makedirs("graficos", exist_ok=True)

# Dataset (ajustar caminho)
dataset = '2015/diabetes_binary_health_indicators_BRFSS2015.csv'
dados = pd.read_csv(dataset, sep=',')

# Função para salvar informações gerais

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

variaveis_verificacao = list(variaveis_quantitativas.keys()) + list(variaveis_ordinais.keys())
salvar_informacoes_gerais_png(dados, variaveis_verificacao)

# Análise descritiva
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

for var, tipo in variaveis_dict.items():
    print(f"\nVariável: {var} ({tipo})")

    if var not in dados.columns:
        print("Variável não encontrada no dataset.")
        continue

    serie = dados[var].dropna()

    if tipo.startswith("Quantitativa"):
        print(serie.describe())
        print("Moda:", serie.mode().values[0] if not serie.mode().empty else "N/A")

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

        plt.figure()
        sns.countplot(x=serie, order=sorted(serie.unique()), palette="pastel")
        plt.title(f"Distribuição de {var}")
        plt.xlabel(var)
        plt.ylabel("Contagem")
        plt.tight_layout()
        plt.savefig(f"graficos/{var}_barras.png")
        plt.close()

        plt.figure()
        counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, colors=sns.color_palette("pastel"))
        plt.title(f"Distribuição de {var}")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(f"graficos/{var}_pizza.png")
        plt.close()

# Análise Exploratória

os.makedirs("graficos", exist_ok=True)

dados_numericos = dados.select_dtypes(include=["number"])

correlacao = dados_numericos.corr()

mascara = np.triu(np.ones_like(correlacao, dtype=bool))

plt.figure(figsize=(15, 10))

sns.heatmap(
    correlacao,
    mask=mascara,
    cmap='RdPu',
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"shrink": .8}
)

plt.title("Mapa de Calor das Correlações", fontsize=18)

plt.tight_layout()
plt.savefig("graficos/heatmap_correlacoes.png", dpi=300)
plt.close()

print("Heatmap salvo em: graficos/heatmap_correlacoes.png")
