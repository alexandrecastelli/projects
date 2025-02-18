#%% Carrega os pacotes

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from functions import descriptive

#%% Carrega os dados titanic

titanic = pd.read_pickle('titanic.pkl')

#%% Verifica os valores ausentes

print(titanic.isnull().sum())

#%% Define a target e as features

X = titanic.drop('survived', axis=1)
y = titanic['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)

#%% Define os parâmetros do grid search

param_grid = {'n_estimators': [50, 100, 150],
              'max_depth': [2, 3],
              'gamma': [0],
              'learning_rate': [0.1, 0.4],
              'colsample_bytree': [0.6, 0.8],
              'min_child_weight': [1],
              'subsample': [0.75, 1]}

#%% Treina o modelo

import time
tempo_ini = time.time()

xgb = XGBClassifier(objective='binary:logistic', random_state=42)

grid_search = GridSearchCV(estimator=xgb, 
                           param_grid=param_grid,
                           scoring='roc_auc', 
                           cv=10, 
                           verbose=0, 
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

tempo_fim = time.time()
print(f"Tempo de execução: {tempo_fim - tempo_ini} segundos")

#%% Avalia o modelo

def evaluate(modelo, X_train, y_train, X_test, y_test):
    p_train = modelo.predict_proba(X_train)[:, 1]
    
    p_test = modelo.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, p_train)
    auc_test = roc_auc_score(y_test, p_test)
    
    print(f'Avaliação base de treino: AUC = {auc_train:.2f}')
    print(f'Avaliação base de teste: AUC = {auc_test:.2f}')
    
    fpr_train, tpr_train, _ = roc_curve(y_train, p_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, p_test)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_train, tpr_train, color='red', label=f'Treino AUC = {auc_train:.2f}')
    plt.plot(fpr_test, tpr_test, color='blue', label=f'Teste AUC = {auc_test:.2f}')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('Falso Positivo')
    plt.ylabel('Verdadeiro Positivo')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()

evaluate(grid_search.best_estimator_, X_train, y_train, X_test, y_test)

#%% Avalia a previsão do modelo

titanic['pred'] = grid_search.best_estimator_.predict_proba(X)[:, 1]

descriptive(titanic, target='survived', features='pred', max_classes=10)