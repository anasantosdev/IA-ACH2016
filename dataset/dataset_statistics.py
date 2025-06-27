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
# - Comando para execução: python3 dataset_statistics.py -d 2023/diabetes_binary_5050split_health_indicators_BRFSS2023.csv
# - Os gráficos serão salvos na pasta ./graficos/ e ./tabelas/
# - Usa dicionário de variáveis importado do arquivo variables-dicitionary.py
# ================================================
# Imports 
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import seaborn as sns
from textwrap import wrap
import io

from variables_dictionary import variaveis_binarias, variaveis_ordinais, variaveis_quantitativas

# ================================================
# Informações gerais

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

# ================================================
# Análise descritiva

def gerar_graficos_descritivos(dados: pd.DataFrame, variaveis_dict: dict, pasta_saida='graficos'):

    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    os.makedirs(pasta_saida, exist_ok=True)

    for var, tipo in variaveis_dict.items():
        print(f"\nVariável: {var} ({tipo})")

        if var not in dados.columns:
            print("Variável não encontrada no dataset.")
            continue

        serie = dados[var].dropna()

        if tipo.lower().startswith("quantitativa"):
            print(serie.describe())
            print("Moda:", serie.mode().values[0] if not serie.mode().empty else "N/A")

            plt.figure()
            sns.histplot(serie, kde=True, bins=20, color="steelblue")
            plt.title(f"Histograma: {var}")
            plt.xlabel(var)
            plt.ylabel("Frequência")
            plt.tight_layout()
            plt.savefig(f"{pasta_saida}/{var}_histograma.png")
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
            plt.savefig(f"{pasta_saida}/{var}_barras.png")
            plt.close()

            # Gráfico de pizza
            plt.figure()
            counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, colors=sns.color_palette("pastel"))
            plt.title(f"Distribuição de {var}")
            plt.ylabel("")
            plt.tight_layout()
            plt.savefig(f"{pasta_saida}/{var}_pizza.png")
            plt.close()

# ================================================
# Análise Cruzada entre variáveis

def plotar_comparacao_grafica(dados, variaveis_binarias, variaveis_ordinais, variaveis_quantitativas):
    os.makedirs("graficos/comparacao_diabetes", exist_ok=True)
    os.makedirs("tabelas", exist_ok=True)
    sns.set(style="whitegrid")

    tabela_binarias = []

    for var in dados.columns:
        if var == "Diabetes_binário" or var not in dados.columns:
            continue

        if var in variaveis_binarias:
            # Gráfico
            plt.figure(figsize=(6, 4))
            sns.barplot(x=var, y="Diabetes_binário", data=dados, ci=None, palette="pastel")
            plt.title(f"Proporção de Diabetes por {var}")
            plt.tight_layout()
            plt.savefig(f"graficos/comparacao_diabetes/{var}_barplot.png")
            plt.close()

            # Tabela de estatísticas para variáveis binárias
            for categoria in [0, 1]:
                subset = dados[dados[var] == categoria]
                total_percent = len(subset) / len(dados) * 100
                if len(subset) > 0:
                    diab_percent = subset["Diabetes_binário"].mean() * 100
                    nao_diab_percent = 100 - diab_percent
                else:
                    diab_percent = nao_diab_percent = 0

                tabela_binarias.append({
                    "Variável": var,
                    "Categoria": categoria,
                    "Porcentagem Total (%)": round(total_percent, 2),
                    "Porcentagem Diabéticos (%)": round(diab_percent, 2),
                    "Porcentagem Não Diabéticos (%)": round(nao_diab_percent, 2)
                })

        elif var in variaveis_ordinais:
            plt.figure(figsize=(8, 4))
            sns.pointplot(x=var, y="Diabetes_binário", data=dados, ci=None, color="steelblue")
            plt.title(f"Proporção de Diabetes por {var}")
            plt.tight_layout()
            plt.savefig(f"graficos/comparacao_diabetes/{var}_pointplot.png")
            plt.close()

        elif var in variaveis_quantitativas:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x="Diabetes_binário", y=var, data=dados, palette="pastel")
            plt.title(f"Distribuição de {var} por Diabetes")
            plt.xticks([0, 1], ["Não diabético", "Diabético"])
            plt.tight_layout()
            plt.savefig(f"graficos/comparacao_diabetes/{var}_boxplot.png")
            plt.close()

    df_tabela_binarias = pd.DataFrame(tabela_binarias)
    df_tabela_binarias.to_csv("tabelas/tabela_variaveis_binarias.csv", index=False)
    print("Tabela com variáveis binárias salva em: tabelas/tabela_variaveis_binarias.csv")

# ================================================
# Análise Exploratória (Correlação)

def gerar_heatmap_correlacoes(df: pd.DataFrame, pasta_saida: str = "graficos", nome_arquivo: str = "heatmap_correlacoes.png"):

    os.makedirs(pasta_saida, exist_ok=True)

    dados_numericos = df.select_dtypes(include=["number"])
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

    caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)
    plt.savefig(caminho_arquivo, dpi=300)
    plt.close()

    print(f"Heatmap salvo em: {caminho_arquivo}")

# ================================================
# Função Main

def main():
    parser = argparse.ArgumentParser(description="Carregar dataset e gerar análises.")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help="Caminho do arquivo CSV do dataset, ex: 2015/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    )
    args = parser.parse_args()

    caminho_dataset = args.dataset

    if not os.path.isfile(caminho_dataset):
        print(f"Arquivo não encontrado: {caminho_dataset}")
        return

    dados = pd.read_csv(caminho_dataset, sep=',')

    variaveis_dict = {}

    for var, props in variaveis_binarias.items():
        variaveis_dict[var] = props["tipo"]
    for var, props in variaveis_ordinais.items():
        variaveis_dict[var] = props["tipo"]
    for var, props in variaveis_quantitativas.items():
        variaveis_dict[var] = props["tipo"]

    os.makedirs("graficos", exist_ok=True)

    variaveis_verificacao = list(variaveis_quantitativas.keys()) + list(variaveis_ordinais.keys())
    salvar_informacoes_gerais_png(dados, variaveis_verificacao)
    gerar_graficos_descritivos(dados, variaveis_dict)
    plotar_comparacao_grafica(dados, variaveis_binarias, variaveis_ordinais, variaveis_quantitativas)
    gerar_heatmap_correlacoes(dados)

if __name__ == "__main__":
    main()