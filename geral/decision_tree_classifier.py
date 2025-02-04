#%% 
# Carrega as bibliotecas

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (accuracy_score, 
                             classification_report,
                             confusion_matrix,
                             balanced_accuracy_score)

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from funcoes_ajuda import descritiva, relatorio_missing

#%%
# Carrega os dados

df = sns.load_dataset('titanic')

#%% 
# Visualiza as primeiras linhas e as colunas do dataset

print(df.head())
print(df.columns)

#%%  
# Mostra a estatística descritiva básica das variáveis

for column in df.columns:
    print(f'\n\nAnálise univariada de {column}:')
    print(df[column].describe())

for column in ['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone']:
    print(f'\n\nFrequências da variável: {column}')
    print(df[column].value_counts(dropna=False).sort_index())

descritiva(df, 'sex')
descritiva(df, 'class')
descritiva(df, 'age', max_classes=10)
descritiva(df, 'fare', max_classes=5)
descritiva(df, 'embarked')
descritiva(df, 'sibsp')
descritiva(df, 'parch')