# carrega as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score
from funcoes_ajuda import descritiva, avalia_clf

# carrega os dados
df = pd.read_parquet('exercicio.parquet')

# verifica parte dos dados
print(df.info())
print(df.head())

# verifica as estatísticas descritivas básicas
for var in df.columns:
    descritiva(df, var=var, vresp = 'inadimplencia')

# seleciona as features e a target
X = df.drop(columns = ['inadimplencia'])
y = df['inadimplencia']

# define as bases de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2360873)

# confere o número de linhas e colunas
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# treina o modelo
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# avalia a base de treino
avalia_clf(clf, y_train, X_train, rótulos_y=['Bom', 'Mau'], base='treino')

# avalia a base de teste
avalia_clf(clf, y_test, X_test, rótulos_y=['Bom', 'Mau'], base='teste')

# cost-complexity pruning path
ccp_path = pd.DataFrame(clf.cost_complexity_pruning_path(X_train, y_train))

# tunando a árvore
GINIs = []

# treino da árvore com diferentes valores de ccp_alpha
for ccp in ccp_path['ccp_alphas']:
    clf = DecisionTreeClassifier(criterion='gini', max_depth=30, random_state=42, ccp_alpha=ccp)

    # treina o modelo
    clf.fit(X_train, y_train)
    AUC = roc_auc_score(y_test, clf.predict_proba(X_test)[:, -1])
    GINI = (AUC-0.5)*2
    GINIs.append(GINI)

# plota o gráfico
sns.lineplot(x = ccp_path['ccp_alphas'], y = GINIs)

# cria o dataframe de avaliações
df_avaliacoes = pd.DataFrame({'ccp': ccp_path['ccp_alphas'], 'GINI': GINIs})

# encontra o melhor parâmetro
GINI_max = df_avaliacoes.GINI.max()
ccp_max  = df_avaliacoes.loc[df_avaliacoes.GINI == GINI_max, 'ccp'].values[0]

# cria o gráfico
plt.ylabel('GINI da árvore')
plt.xlabel('CCP Alphas')
plt.title('Avaliação da árvore por valor de CCP-Alpha')
plt.show()

# mostra o melhor parâmetro
print(f'O GINI máximo é de: {GINI_max:.2%}\nObtido com um ccp de: {ccp_max}')

# árvore ótima
clf_tuned = DecisionTreeClassifier(criterion='gini',
                                max_depth = 30, 
                                random_state=42,
                                ccp_alpha=ccp_max).fit(X_train, y_train)

# avalia a base de treino
print('Avaliando a base de treino:')
avalia_clf(clf_tuned, y_train, X_train, base='treino')

# avalia a base de teste
print('Avaliando a base de teste:')
avalia_clf(clf_tuned, y_test, X_test, base='teste')

# plota a árvore
plt.figure(figsize=(20, 10))
plot_tree(clf_tuned, feature_names=X.columns.tolist(), class_names=['Bom', 'Mau'], filled=True)
plt.show()