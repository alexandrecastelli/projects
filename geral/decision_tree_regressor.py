#%%

# carrega as bibliotecas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score

#%%

# define a seed
np.random.seed(2360873)

# gera 1000 valores para X entre 0 e 1
x = np.linspace(0, 1, 1000)

# define os parâmetros da parábola
a = 0
b = 10
c = -10

# gera uma relação quadrática com ruído
y = a + b * x + c * x**2 + np.random.normal(loc=0, scale=.3, size=len(x))**3

# cria o data frame
df = pd.DataFrame({'x':x, 'y':y})

# gera o gráfico
plt.figure(figsize=(10, 6))
sns.scatterplot(x='x', y='y', data=df, color='skyblue', label='Observado')

# ajusta e mostra o gráfico
plt.title('Relação quadrática com ruído')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.show()

#%%

# cria uma função que mostra os valores observados, os previstos e os resíduos de acordo com a profundidade da árvore

def tree_function(depth=3, alpha=0):

    # define a árvore
    tree = DecisionTreeRegressor(max_depth=depth, ccp_alpha=alpha) 
    tree.fit(df[['x']], df['y'])
    
    # gera os valores previstos e os resíduos
    df['p'] = tree.predict(df[['x']])
    df['r'] = df['y'] - df['p']
    
    # cria um gráfico com 2 subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # gera o gráfico dos valores observados e previstos
    sns.scatterplot(x='x', y='y', data=df, color='skyblue', label='Observado', ax=ax[0])
    ax[0].plot(df['x'], df['p'], color='red', label='Predito')
    ax[0].set_title(f'Observados vs Esperados - profundidade = {depth}')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].legend()
    
    # gera o gráficos dos resíduos
    sns.scatterplot(x='x', y='r', data=df, color='skyblue', label='Resíduos', ax=ax[1])
    ax[1].set_title(f'Resíduos com profundidade = {depth}')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('Resíduos')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

#%%

# excuta a função para diferentes profundidades
for depth in [1, 2, 3, 5, 10, 30]:
    tree_function(depth)
    
#%%

# transforma os arrays em dataframes
x = df[['x']]
y = df[['y']]

# define as bases de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#%%

# define a árvore de decisão
tree = DecisionTreeRegressor(max_depth=30, ccp_alpha=0)

# treina a árvore
tree.fit(X_train, y_train)

#%%

# gera os valores do cost-complexity pruning path
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# gera o gráfico de ccp_alphas e impurities
sns.scatterplot(x = ccp_alphas, y = impurities)

# mostra a quantidade de valores de ccp_alphas
len(ccp_alphas)

#%%

# cria um dicionário com valores de ccp_alpha
param_grid  = {'ccp_alpha': ccp_alphas[::10]}

# define o grid search
gs_tree = GridSearchCV(estimator=tree,
                           param_grid=param_grid,
                           cv=5,
                           scoring='neg_mean_squared_error')

# treina o modelo com o grid search
gs_tree.fit(X_train, y_train)

#%%

# obtém os melhores parâmetros
best_params = gs_tree.best_params_
print(best_params)

#%%

# define o modelo com os melhores parâmetros
best_tree = DecisionTreeRegressor(**best_params, max_depth=30)
best_tree.fit(X_train, y_train)

#%%

# avalia o modelo na base de teste
y_pred = best_tree.predict(X_test)

# calcula o R-quadrado
r2 = r2_score(y_test, y_pred)
print("R-quadrado na base de testes:", r2)

#%%

# gera o gráfico com os valores observados e previstos
plt.scatter(y_test, y_pred)
plt.xlabel('Valores reais')
plt.ylabel('Valores previstos')
plt.title('Comparação entre valores reais e previstos')
plt.show()

#%%

# mostra a árvore com os melhotres parâmetros
plt.figure(figsize=(10, 6))
plot_tree(best_tree, filled=True, feature_names=['x'])
plt.show()

# gera o gráfico com os valores observados, os previstos e os resíduos da árvore com os melhores parâmetros
tree_function(depth=30, alpha=best_params['ccp_alpha'])