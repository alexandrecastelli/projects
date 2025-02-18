#%% Carrega os pacotes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Gera os dados

np.random.seed(42)
df = pd.DataFrame({'dados': np.random.normal(size=100)})

#%% Calcula o erro padrão por bootstrap

def function_sem(data, n_boot=1000):
  n = len(data)
  mean = np.zeros(n_boot)
  for i in range(n_boot):
    bootstrap_sample = np.random.choice(data, size=n, replace=True)
    mean[i] = np.mean(bootstrap_sample)
  return mean

#%% Calcula o erro padrão da média

bootstrap_sample = function_sem(df['dados'])

#%% Gera o histograma e calcula o desvio padrão das médias

plt.hist(bootstrap_sample)
plt.title('Histograma das médias')
plt.xlabel('Médias')
plt.ylabel('Frequência')
plt.show()

stddv = np.std(bootstrap_sample)
print(f"Desvio padrão das médias: {stddv}")