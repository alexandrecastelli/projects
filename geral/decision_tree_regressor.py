#%% Carrega os pacotes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score

#%% Define a parábola com ruído

np.random.seed(42)

x = np.linspace(0, 1, 1000)

a = 0
b = 10
c = -10

y = a + b * x + c * x**2 + np.random.normal(loc=0, scale=.3, size=len(x))**3

#%% Cria o data frame

df = pd.DataFrame({'x':x, 'y':y})

#%% Gera o gráfico

plt.figure(figsize=(10, 6))
sns.scatterplot(x='x', y='y', data=df, color='skyblue', label='Observado')

plt.title('Relação quadrática com ruído')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.show()

#%% Cria uma função que analisa o modelo

def tree_function(depth=3, alpha=0):

    tree = DecisionTreeRegressor(max_depth=depth, ccp_alpha=alpha, random_state=42) 
    tree.fit(df[['x']], df['y'])
    
    df['p'] = tree.predict(df[['x']])
    df['r'] = df['y'] - df['p']
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    sns.scatterplot(x='x', y='y', data=df, color='skyblue', label='Observado', ax=ax[0])
    ax[0].plot(df['x'], df['p'], color='red', label='Predito')
    ax[0].set_title(f'Observados vs Esperados - profundidade = {depth}')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].legend()
    
    sns.scatterplot(x='x', y='r', data=df, color='skyblue', label='Resíduos', ax=ax[1])
    ax[1].set_title(f'Resíduos com profundidade = {depth}')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('Resíduos')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

#%% Executa a função para diferentes profundidades

for depth in [1, 2, 3, 5, 10, 30]:
    tree_function(depth)
    
#%% Transforma os arrays em dataframes

x = df[['x']]
y = df[['y']]

#%% Treina o modelo

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

tree = DecisionTreeRegressor(max_depth=30, ccp_alpha=0, random_state=42)

tree.fit(X_train, y_train)

#%% Gera os valores do cost-complexity pruning path

path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

#%% Gera o gráfico de ccp_alphas e impurities

sns.scatterplot(x = ccp_alphas, y = impurities)

len(ccp_alphas)

#%% Cria um dicionário com valores de ccp_alpha

param_grid  = {'ccp_alpha': ccp_alphas[::10]}

# Define o grid search

gs_tree = GridSearchCV(estimator=tree,
                           param_grid=param_grid,
                           cv=5,
                           scoring='neg_mean_squared_error')

gs_tree.fit(X_train, y_train)

#%% Obtém os melhores parâmetros

best_params = gs_tree.best_params_
print(best_params)

#%% Define o modelo com os melhores parâmetros

best_tree = DecisionTreeRegressor(**best_params, max_depth=30, random_state=42)
best_tree.fit(X_train, y_train)

#%% Avalia o modelo na base de teste

y_pred = best_tree.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("R-quadrado na base de testes:", r2)

#%% Gera o gráfico com os valores observados e previstos

plt.scatter(y_test, y_pred)
plt.xlabel('Valores reais')
plt.ylabel('Valores previstos')
plt.title('Comparação entre valores reais e previstos')
plt.show()

#%% Mostra a árvore com os melhotres parâmetros

plt.figure(figsize=(10, 6))
plot_tree(best_tree, filled=True, feature_names=['x'])
plt.show()

tree_function(depth=30, alpha=best_params['ccp_alpha'])