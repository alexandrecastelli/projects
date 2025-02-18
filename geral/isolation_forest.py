#%% Carrega as bibliotecas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

#%% Gera os dados normais

np.random.seed(42)
normal_data = np.random.randn(1000, 2)

#%% Adiciona outliers

outliers = np.random.uniform(low=-5, high=5, size=(20, 2))

#%% Concatena tudo

data = np.vstack((normal_data, outliers))

#%% Normaliza os dados

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

#%% Converte para data frame

df = pd.DataFrame(data_scaled, columns=['X1', 'X2'])

#%% Plota os dados

plt.scatter(df['X1'], df['X2'], label="Dados Normais")
plt.scatter(outliers[:, 0], outliers[:, 1], color='red', label="Outliers")
plt.legend()
plt.show()

#%% Cria o modelo Isolation Forest

iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)

#%% Treina e prevê anomalias (-1 = outlier, 1 = normal)

df['Anomaly_Score'] = iso_forest.fit_predict(data_scaled)

#%% Converte para booleano

df['Outlier'] = df['Anomaly_Score'] == -1

#%% Plota os resultados

plt.scatter(df['X1'], df['X2'], c=df['Outlier'], cmap='coolwarm', label="Outliers Detectados")
plt.legend()
plt.show()

#%% Mostra os primeiros outliers detectados
print(df[df['Outlier']])