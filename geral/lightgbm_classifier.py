#%% Carrega os pacotes

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
import lightgbm as lgb
import time
import numpy as np

#%% Carrega os dados

X_train = pd.read_pickle('X_train.pkl')
y_train = pd.read_pickle('y_train.pkl')
X_test = pd.read_pickle('X_test.pkl')
y_test = pd.read_pickle('y_test.pkl')

duplicated_columns = X_train.columns[X_train.columns.duplicated()]
print(f'Colunas duplicadas: {duplicated_columns}')
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]

#%% Ajusta o índice

X_train.set_index('subject', append=True, inplace=True)
X_test.set_index('subject', append=True, inplace=True)

#%% Adiciona a coluna de resposta no data frame de treino

HAR_train = pd.concat([X_train.reset_index(), y_train], axis=1).set_index(['level_0', 'subject'])

#%% Visualiza as colunas

print(HAR_train.columns)

#%% Analisa as estatísticas

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(data=HAR_train, x='label', y=HAR_train.iloc[:, 1], ax=axs[0])
axs[0].set_xlabel("Atividade realizada")
axs[0].set_ylabel(HAR_train.columns[1])
axs[0].set_title("Aceleração média(x) por atividade")
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=30)

sns.boxplot(data=HAR_train, x='label', y=HAR_train.iloc[:, 3], ax=axs[1])
axs[1].set_xlabel("Atividade realizada")
axs[1].set_ylabel(HAR_train.columns[3])
axs[1].set_title("Aceleração média(x) por atividade")
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=30)

sns.boxplot(data=HAR_train, x='label', y=HAR_train.iloc[:, 14], ax=axs[2])
axs[2].set_xlabel("Atividade realizada")
axs[2].set_ylabel(HAR_train.columns[14])
axs[2].set_title("Aceleração média(x) por atividade")
axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=30)

plt.show()

#%% Treina o modelo

np.random.seed(42)
tempo_ini = time.time()
arvore = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=2)
arvore.fit(X_train, y_train)
tempo_fim = time.time()
print(f'Tempo de treinamento da árvore: {tempo_fim - tempo_ini} segundos')

importancias = pd.DataFrame(arvore.feature_importances_, index=X_train.columns, columns=['importancia'])
top_10_variaveis = importancias.sort_values(by='importancia', ascending=False)[:10]
print(f'Top 10 variaveis: {top_10_variaveis}')

variaveis = importancias.nlargest(20, 'importancia').index.tolist()
print(f'Variáveis selecionadas: {variaveis}')

param_grid = {'num_leaves': [31],
              'max_depth': [3, 10],
              'learning_rate': [0.05, 0.2],
              'n_estimators': [5, 11]}

#%% Verifica os valores faltantes

print(X_train[variaveis].isna().sum())
print(y_train.isna().sum())

#%% Prepara a variável y

y = y_train['label'].cat.codes

#%% Treina o modelo

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')

tempo_ini = time.time()

modelo = lgb.LGBMClassifier(objective='multiclass', random_state=42)

grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, scoring=scorer, cv=cv, n_jobs=-1, verbose=1)
grid_search.fit(X_train[variaveis], y)

tempo_fim = time.time()
print(f'Tempo de treinamento do Grid Search: {tempo_fim - tempo_ini} segundos')

print(f'Melhores Parâmetros: {grid_search.best_params_}')

#%% Mostra os resultados

resultados_cv = pd.DataFrame(grid_search.cv_results_)

pred_test = pd.Series(grid_search.best_estimator_.predict(X_test[variaveis]))

print(pd.crosstab(pred_test, y_test.label))

acurácia = (pred_test == y_test.label.cat.codes).sum()/len(y_test)

print(f'acurácia = {acurácia:.2%}')