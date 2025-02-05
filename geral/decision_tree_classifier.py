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

from functions import descriptive, missing

#%%
# Carrega os dados e visualiza informações do dataset

df = sns.load_dataset('titanic')

print(df.info())
print(df.head())

#%%  
# Visualiza estatísticas das variáveis

for column in df.columns:
    print(f'\n\nAnálise univariada de {column}:')
    print(df[column].describe())

for column in ['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone']:
    print(f'\n\nFrequências da variável: {column}')
    print(df[column].value_counts(dropna=False).sort_index())

descriptive(df, 'survived', 'sex') # incluir um loop na função
descriptive(df, 'survived', 'class')
descriptive(df, 'survived', 'age')
descriptive(df, 'survived', 'fare')
descriptive(df, 'survived', 'embarked')
descriptive(df, 'survived', 'sibsp')
descriptive(df, 'survived', 'parch')

#%%
# Analisa os dados faltantes

missing(df)

# %%
# Trata a variável age e remove as variáveis redundantes

df['age'] = df.age.fillna(df.age.mean())

df.drop(columns=['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone'], inplace=True)

#%% 
# Transforma as variáveis strings em dummies e salva a base tratada

df_dummies = pd.get_dummies(df, drop_first=True)
print(df_dummies.info())
print(df_dummies.head())

df_dummies.to_pickle('titanic.pkl')

#%%  
# Define a target, as features e os hiperparâmetros e treina do modelo

y = df_dummies['survived']
X = df_dummies.drop(columns = ['survived'])

clf = DecisionTreeClassifier(criterion='gini', max_depth = 3, random_state=42)

clf.fit(X, y)

#%%  
# Visualiza a árvore

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns.tolist(), class_names=['Not Survived', 'Survived'], filled=True)
plt.show()

#%%
# Classifica novos dados

df_new = X.tail()
print(df_new)

clf_new = clf.predict(df_new)
clf_new

#%%  
# Avalia a classificação

clf_train = clf.predict(X)

print(pd.crosstab(clf_train, y, margins=True))
print(pd.crosstab(clf_train, y, normalize='index'))
print(pd.crosstab(clf_train, y, normalize='columns'))

hits = clf_train == y
hits_pct = hits.sum()/hits.shape[0]
print(f'Acurácia (taxa de acertos): {hits_pct:.2%}')

#%% 

# Cria a matriz de confusão e calcula a acurácia e a acurácia balanceada (força uma distribuição uniforme para a target)

cm = confusion_matrix(y, clf.predict(X))
ac = accuracy_score(y, clf.predict(X))
bac = balanced_accuracy_score(y, clf.predict(X))

print(f'\nA acurácia da árvore é: {ac:.1%}')
print(f'A acurácia balanceada da árvore é: {bac:.1%}')

#%%
# Visualiza o mapa de calor da matriz de confusão

sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
            xticklabels=['Não Sobreviveu', 'Sobreviveu'], 
            yticklabels=['Não Sobreviveu', 'Sobreviveu'])
plt.show()

print('\n', classification_report(y, clf.predict(X)))