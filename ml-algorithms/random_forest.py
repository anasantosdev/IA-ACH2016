import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Carregando dados
df = pd.read_csv('data/diabetes.csv')

# Selecionando variáveis
variables = ['HighBP', 'BMI', 'Age', 'Sex', 'HighChol']
X= df[variables]
y = df['Diabetes_binary']

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Criando e treinando a Ramdom Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)


y_pred = modelo.predict(X_test)
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

#TODO: Verificar como arrumar o modelo ao longo dos testes
#TODO: Verificar como sabre o  que é glicose alta, etc.
#TODO: Ver o que inclui essa avalição de modelo, 
# porque queremos matriz de confusão para verificação de acurácia
# Avaliação do modelo 
