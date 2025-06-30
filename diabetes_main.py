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
# - Exemplo de uso: python3 diabetes_main.py -d dataset/2023/diabetes_binary_5050split_health_indicators_BRFSS2023.csv -a logistic
# ================================================
# Imports 

import os
import argparse
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# ================================================
# Métodos para ML

def carregar_dataset(caminho_csv):
    df = pd.read_csv(caminho_csv)
    return df

def preprocessar_dataset(df, target="Diabetes_binário"):
    variaveis_nao_binarias = [
        "IMC", "Saúde_Geral", "Idade", "Nível_Educação", "Renda"
    ]

    # Variáveis com baixa significância estatística
    variaveis_baixo_impacto = [
        "Acesso_Saúde", "Fumante", "Seguro_Saúde", "Saúde_Mental", 
        "Saúde_Física", "Avaliou_Colesterol", "Consumo_Álcool"
    ]
    
    X = df.drop(columns=[target] + variaveis_baixo_impacto)
    y = df[target]

    # Padronização das variáveis não binárias
    colunas_validas = [col for col in variaveis_nao_binarias if col in X.columns]

    scaler = StandardScaler()
    X[colunas_validas] = scaler.fit_transform(X[colunas_validas])

    # Data splitting 
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def treinar_modelo(algoritmo, X_train, y_train):
    if algoritmo == "logistic":
        modelo = LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
    elif algoritmo == "tree":
        modelo = DecisionTreeClassifier(
            criterion="gini",        
            max_depth=10,            
            min_samples_split=50,    
            min_samples_leaf=25,    
            random_state=42
        )
    elif algoritmo == "forest":
        modelo = RandomForestClassifier(
            n_estimators=300,         
            max_depth=None,           
            max_features="sqrt",      
            min_samples_split=5,      
            min_samples_leaf=2,       
            random_state=42,
            n_jobs=-1
        )
    elif algoritmo == "gradient":
        modelo = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    else:
        raise ValueError(f"Algoritmo desconhecido: {algoritmo}")
    
    modelo.fit(X_train, y_train)

    return modelo

def avaliar_modelo(modelo, X, y, nome_conjunto="validação", salvar_em="avaliacoes"):
    y_pred = modelo.predict(X)
    acc = accuracy_score(y, y_pred)
    relatorio = classification_report(y, y_pred)
    matriz = confusion_matrix(y, y_pred)

    if hasattr(modelo, "predict_proba"):
        y_scores = modelo.predict_proba(X)[:, 1]  
        auc = roc_auc_score(y, y_scores)
        fpr, tpr, _ = roc_curve(y, y_scores)
    else:
        y_scores = None
        auc = None
        fpr, tpr = None, None

    print(f"\n======= Acurácia no conjunto de {nome_conjunto}: {acc:.4f}")
    print(f"\n======= Relatório de classificação ({nome_conjunto}):")
    print(relatorio)
    print(f"\n======= Matriz de confusão ({nome_conjunto}):")
    print(matriz)
    if auc is not None:
        print(f"\n======= AUC-ROC ({nome_conjunto}): {auc:.4f}")

    os.makedirs(salvar_em, exist_ok=True)

    caminho_relatorio = os.path.join(salvar_em, f"relatorio_{nome_conjunto}.txt")
    with open(caminho_relatorio, "w") as f:
        f.write(f"Acurácia: {acc:.4f}\n")
        if auc is not None:
            f.write(f"AUC-ROC: {auc:.4f}\n")
        f.write("\nRelatório de classificação:\n")
        f.write(relatorio)
        f.write("\nMatriz de confusão:\n")
        f.write(str(matriz))

    print(f"\n======= Relatório salvo em: {caminho_relatorio}")

    plt.figure(figsize=(6, 4))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Matriz de Confusão ({nome_conjunto})")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    caminho_img = os.path.join(salvar_em, f"matriz_confusao_{nome_conjunto}.png")
    plt.savefig(caminho_img)
    print(f"\n======= Matriz de confusão salva em: {caminho_img}")
    plt.close()

    if auc is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("FPR (Falsos Positivos)")
        plt.ylabel("TPR (Verdadeiros Positivos)")
        plt.title(f"Curva ROC ({nome_conjunto})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        caminho_roc = os.path.join(salvar_em, f"curva_roc_{nome_conjunto}.png")
        plt.savefig(caminho_roc)
        print(f"\n======= Curva ROC salva em: {caminho_roc}")
        plt.close()

def interpretar_coefs(modelo, X_train, salvar_em="avaliacoes"):
    coefs = modelo.coef_[0]
    variaveis = X_train.columns

    print(f"\n{'Variável':20} | {'Coeficiente':12} | {'Odds Ratio':10}")
    print("-" * 50)
    for var, coef in zip(variaveis, coefs):
        oratio = np.exp(coef)
        print(f"{var:20} | {coef:+.4f}       | {oratio:.4f}")

    os.makedirs(salvar_em, exist_ok=True)

    df_coefs = pd.DataFrame({
        "Variável": variaveis,
        "Coeficiente": coefs,
        "Odds Ratio": np.exp(coefs)
    }).sort_values(by="Coeficiente", key=abs, ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(
        x="Coeficiente", 
        y="Variável", 
        data=df_coefs, 
        hue="Variável", 
        palette="coolwarm", 
        legend=False
    )
    plt.title("Coeficientes da Regressão Logística")
    plt.tight_layout()

    caminho_img = os.path.join(salvar_em, "coeficientes_logistic_regression.png")
    plt.savefig(caminho_img)
    plt.close()

    print(f"\n======= Gráfico de coeficientes salvo em: {caminho_img}")

def interpretar_feature_importance(modelo, X_train, salvar_em="avaliacoes"):
    importancias = modelo.feature_importances_
    variaveis = X_train.columns

    df_imp = pd.DataFrame({"Variável": variaveis, "Importância": importancias})
    df_imp = df_imp.sort_values(by="Importância", ascending=False)

    print(df_imp)

    plt.figure(figsize=(8,6))
    sns.barplot(x="Importância", y="Variável", data=df_imp)
    plt.title("Importância das Variáveis")
    plt.tight_layout()
    caminho_img = os.path.join(salvar_em, "coeficientes_modelo.png")
    plt.savefig(caminho_img)
    plt.close()

    print(f"\n======= Gráfico de coeficientes salvo em: {caminho_img}")

# ================================================
# Função Main

def main():
    parser = argparse.ArgumentParser(description="\n======= Preditor de Diabetes com ML =======")
    parser.add_argument("-d", "--dataset", required=True, help="Caminho para o dataset CSV")
    parser.add_argument("-a", "--algoritmo", required=True, choices=["logistic", "tree", "forest", "gradient"], help="Algoritmo ML a ser usado")
    
    args = parser.parse_args()
    
    print(f"\n======= Lendo dados de: {args.dataset}")
    df = carregar_dataset(args.dataset)

    print("\n======= Pré-processando dados...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessar_dataset(df)

    print("\nX_train (5 primeiros):")
    print(X_train.head())

    print("\nX_val (5 primeiros):")
    print(X_val.head())

    print("\nX_test (5 primeiros):")
    print(X_test.head())

    print("\ny_train (5 primeiros):")
    print(y_train.head())

    print("\ny_val (5 primeiros):")
    print(y_val.head())

    print("\ny_test (5 primeiros):")
    print(y_test.head())

    print(f"\n======= Treinando modelo: {args.algoritmo}")
    modelo = treinar_modelo(args.algoritmo, X_train, y_train)

    print(f"\n======= Gerando avaliação do modelo: {args.algoritmo}")
    avaliar_modelo(modelo, X_val, y_val, nome_conjunto="validação")
    avaliar_modelo(modelo, X_test, y_test, nome_conjunto="teste")

    if args.algoritmo == "logistic":
        print("\n======= Interpretando coeficientes do modelo:")
        interpretar_coefs(modelo, X_train)
    else:
        print("\n======= Importância das variáveis (feature importance):")
        interpretar_feature_importance(modelo, X_train)

    print("\n======= Fim da Execução =======")

if __name__ == "__main__":
    main()
