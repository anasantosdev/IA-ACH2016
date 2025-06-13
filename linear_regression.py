import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar dados
df = pd.read_csv('data/diabetes.csv')
variables = ['HighBP', 'BMI', 'Age', 'Sex', 'HighChol']
X= df[variables]
y = df['Diabetes_binary']

X = df.drop(columns=['Diabetes_binary']).values
y = df['Diabetes_binary'].values.reshape(-1, 1)

# Normalizar dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Converter para tensores
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Modelo de regressão linear com ativação sigmoid
class DiabetesModel(nn.Module):
    def __init__(self, input_dim):
        super(DiabetesModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Instanciar modelo
input_dim = X_train.shape[1]
model = DiabetesModel(input_dim)

# Função de perda e otimizador
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Treinamento
for epoch in range(1000):
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Avaliação
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    y_pred_labels = (y_pred_test > 0.5).float()

# Métricas
from sklearn.metrics import classification_report
print("Relatório de Classificação:\n", classification_report(y_test, y_pred_labels))