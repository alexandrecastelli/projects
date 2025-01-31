# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 22:26:07 2025

@author: João Mello
"""
#%% Ler bibliotecas
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from funcoes_ajuda import avalia_clf

#%% LIÇÃO DE CASA

# Lição de casa - Tente ajustar um modelo, com o que você aprendeu, nas bases UCI HAR

X_train = pd.read_pickle('X_train.pkl')
X_test = pd.read_pickle('X_test.pkl')
y_train = pd.read_pickle('y_train.pkl')
y_test = pd.read_pickle('y_test.pkl')


#%%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#%% Montando a estrutura para um gridsearch
param_grid = {'n_estimators': [100], 'max_features': range(1, 10)}

rf_model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf_model, 
                           param_grid=param_grid, 
                           scoring='roc_auc', 
                           cv=4, 
                           n_jobs=-1)

grid_search.fit(X_train, y_train.values.ravel()) 

# Print the best parameters and the best score
print(grid_search)
print(grid_search.best_params_)
print(grid_search.best_score_)

best_model = grid_search.best_estimator_

#%% Avaliar o modelo tunado
# aval_classificador(y_train, X_train, y_test, X_test, best_model)
avalia_clf(best_model, y_train, X_train, base='treino')
avalia_clf(best_model, y_test, X_test, base='teste')