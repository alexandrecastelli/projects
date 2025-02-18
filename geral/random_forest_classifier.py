#%% Carrega os pacotes

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from functions import evaluate
import patsy

#%% Carrega a base de dados

titanic = sns.load_dataset('titanic')
titanic.head()

#%% Cria a matriz de dados com a target e as features usando o pacote patsy

y, X = patsy.dmatrices('survived ~ pclass + sex + age + sibsp + parch + fare + embarked', data=titanic, return_type='dataframe')

print(X.head())
print(y.head())

#%% Divide os dados em conjuntos de treinamento e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

#%% Treina o modelo

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

#%% Cria o data frame de avaliação de treino e teste

evaluate(rf, y_train, X_train, rótulos_y=['Not survived', 'Survived'], base = 'treino')
evaluate(rf, y_test, X_test, rótulos_y=['Not survived', 'Survived'], base = 'teste')

#%% Montando a estrutura para um grid search

param_grid = {'n_estimators': [100], 'max_features': range(1, 11)}

rf_model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf_model, 
                           param_grid=param_grid, 
                           scoring='roc_auc', 
                           cv=4, 
                           n_jobs=-1)

grid_search.fit(X_train, y_train) 

print(grid_search)
print(grid_search.best_params_)
print(grid_search.best_score_)

rf_best = grid_search.best_estimator_

#%% Avalia o modelo otimizado

evaluate(rf_best, y_train, X_train, rótulos_y=['Não Sobreviveu', 'Sobreviveu'], base = 'treino')
evaluate(rf_best, y_test, X_test, rótulos_y=['Não Sobreviveu', 'Sobreviveu'], base = 'teste')