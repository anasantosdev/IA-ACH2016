# ================================================
# Arquivo diabetes-classifier.py
#
# Descrição:
# -  Arquivo principal do projeto AI for Good
# 
# Uso:
# Execute com Python 3.10 ou superior
#
# Observações:
# - 
# ================================================
# Imports 

import argparse
import pandas as pd

from algoritmos_ml.regressao_logistica import LogisticRegression
from algoritmos_ml.arvore_decisao import DecisionTreeClassifier
from algoritmos_ml.floresta_aleatoria import RandomForestClassifier

# ================================================
# Métodos para ML

def carregar_dataset(caminho_csv):
    df = pd.read_csv(caminho_csv)
    return df

def preprocessar_dataset(df, target="Diabetes_binário"):
    X = df.drop(columns=[target])
    y = df[target]
    
    return 

def treinar_modelo(algoritmo, X_train, X_test, y_train, y_test):
    if algoritmo == "logistic":
        modelo = LogisticRegression()
    elif algoritmo == "tree":
        modelo = DecisionTreeClassifier()
    elif algoritmo == "forest":
        modelo = RandomForestClassifier()
    else:
        raise ValueError(f"Algoritmo desconhecido: {algoritmo}")

#TODO: ajustar método abaixo
def avaliar_modelo(algoritmo):
    if algoritmo == "logistic":
        modelo = LogisticRegression()
    elif algoritmo == "tree":
        modelo = DecisionTreeClassifier()
    elif algoritmo == "forest":
        modelo = RandomForestClassifier()
    else:
        raise ValueError(f"Não foi possível avaliar o modelo: {algoritmo}")

# ================================================
# Função Main

def main():
    parser = argparse.ArgumentParser(description="======= Preditor de Diabetes com ML =======")
    parser.add_argument("-d", "--dataset", required=True, help="Caminho para o dataset CSV")
    parser.add_argument("-a", "--algoritmo", required=True, choices=["logistic", "tree", "forest"], help="Algoritmo ML a ser usado")
    
    args = parser.parse_args()
    
    print(f"Lendo dados de: {args.dataset}")
    df = carregar_dataset(args.dataset)

    print("Pré-processando dados...")
    X_train, X_test, y_train, y_test = preprocessar_dataset(df)

    print(f"Treinando modelo: {args.algoritmo}")
    treinar_modelo(args.algoritmo, X_train, X_test, y_train, y_test)

    #TODO: Adicionar código
    print(f"Gerando avaliação do modelo: {args.algoritmo}")
    avaliar_modelo(args.algoritmo)

    print("======= Fim da Execução =======")

if __name__ == "__main__":
    main()
