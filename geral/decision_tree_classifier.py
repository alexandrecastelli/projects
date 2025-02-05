#%% 
# Carrega as bibliotecas

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from functions import descriptive, missing, evaluate

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

for column in df.columns:
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

df = pd.get_dummies(df, drop_first=True)

print(df.info())
print(df.head())

df.to_pickle('titanic.pkl')

#%%  
# Define a target, as features e as bases de teste e de treino

y = df['survived']
X = df.drop(columns=['survived'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#%% 
# Define, treina e avalia o modelo

clf = DecisionTreeClassifier(criterion='gini', 
                             max_depth = 3, 
                             random_state=42)

clf.fit(X_train, y_train)

evaluate(clf, y_train, X_train, base='treino')
evaluate(clf, y_test, X_test, base='teste')

#%% 
# Define, treina e avalia o modelo

clf = DecisionTreeClassifier(criterion='gini', 
                             max_depth = 30,
                             random_state=42,
                             ccp_alpha=0)

# Treinar o modelo
clf.fit(X_train, y_train)

evaluate(clf, y_train, X_train, base='treino')
evaluate(clf, y_test, X_test, base='teste')

#%%
# Define o Cost-Complexity Pruning Path para otimizar o modelo

ccp_path = pd.DataFrame(clf.cost_complexity_pruning_path(X_train, y_train))

#%% 
# Otimiza o modelo

GINIs = []

for ccp in ccp_path['ccp_alphas']:
    clf = DecisionTreeClassifier(criterion='gini', 
                                 max_depth = 30,
                                 random_state=42,
                                 ccp_alpha=ccp)

    clf.fit(X_train, y_train)
    AUC = roc_auc_score(y_test, clf.predict_proba(X_test)[:, -1])
    GINI = (AUC-0.5)*2
    GINIs.append(GINI)

sns.lineplot(x=ccp_path['ccp_alphas'], y=GINIs)

df_avaliacoes = pd.DataFrame({'ccp':ccp_path['ccp_alphas'], 'GINI':GINIs})

GINI = df_avaliacoes.GINI.max()
ccp  = df_avaliacoes.loc[df_avaliacoes.GINI==GINI, 'ccp'].values[0]

plt.ylabel('GINI da árvore')
plt.xlabel('CCP Alphas')
plt.title('Avaliação da árvore por valor de CCP-Alpha')

print(f'O GINI máximo é de: {GINI:.2%}\nObtido com um CCP de: {ccp}')

#%% Define e avalia o modelo otimizado

clf = DecisionTreeClassifier(criterion='gini', 
                             max_depth = 30,
                             random_state=42,
                             ccp_alpha=ccp)

clf.fit(X_train, y_train)

evaluate(clf, y_train, X_train, base='treino')
evaluate(clf, y_test, X_test, base='teste')

#%%  
# Visualiza a árvore

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X_test.columns.tolist(), class_names=['Not survived', 'Survived'], filled=True)
plt.show()