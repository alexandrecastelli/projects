import yfinance as yf
import numpy as np
import pandas as pd
import torch
from transformers import TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 📌 1. BAIXAR DADOS DA B3 (Exemplo: PETR4)
ticker = "PETR4.SA"
df = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# 📌 2. CRIAR FEATURES PARA TREINAMENTO
df["Return"] = df["Adj Close"].pct_change()
df["Volatility"] = df["Return"].rolling(window=10).std()
df["SMA_10"] = df["Adj Close"].rolling(window=10).mean()
df.dropna(inplace=True)  # Remover valores NaN

# 📌 3. PREPARAR OS DADOS PARA O MODELO
features = ["Adj Close", "Return", "Volatility", "SMA_10"]
target = "Adj Close"

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

X = df_scaled[:-1]
y = df_scaled[1:, 0]  # Próximo preço de fechamento como target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Converter para tensores do PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 📌 4. DEFINIR E TREINAR O MODELO TRANSFORMER
config = TimeSeriesTransformerConfig(prediction_length=1, context_length=10, input_size=len(features))
model = TimeSeriesTransformerForPrediction(config)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5  # Para testes, pode ser aumentado
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch.unsqueeze(1)).logits.squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Época {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

# 📌 5. PREVER VALORES FUTUROS
with torch.no_grad():
    predictions = model(X_test_tensor.unsqueeze(1)).logits.squeeze().numpy()

# 📌 6. ANALISAR OS RESULTADOS
plt.figure(figsize=(12, 5))
plt.plot(y_test, label="Real")
plt.plot(predictions, label="Previsto")
plt.legend()
plt.title(f"Previsão de {ticker}")
plt.show()
