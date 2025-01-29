# carrega as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# confere o número de linhas e colunas
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# treina o modelo
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# cost-complexity pruning path
ccp_path = pd.DataFrame(clf.cost_complexity_pruning_path(X_train, y_train))

# tunando a árvore
GINIs = []

for ccp in ccp_path['ccp_alphas']:
    clf = DecisionTreeClassifier(criterion='gini', max_depth=30, random_state=42, ccp_alpha=ccp)

    # treina o modelo
    clf.fit(X_train, y_train)
    AUC = roc_auc_score(y_test, clf.predict_proba(X_test)[:, -1])
    GINI = (AUC-0.5)*2
    GINIs.append(GINI)

sns.lineplot(x = ccp_path['ccp_alphas'], y = GINIs)

df_avaliacoes = pd.DataFrame({'ccp': ccp_path['ccp_alphas'], 'GINI': GINIs})

GINI_max = df_avaliacoes.GINI.max()
ccp_max  = df_avaliacoes.loc[df_avaliacoes.GINI == GINI_max, 'ccp'].values[0]

plt.ylabel('GINI da árvore')
plt.xlabel('CCP Alphas')
plt.title('Avaliação da árvore por valor de CCP-Alpha')

print(f'O GINI máximo é de: {GINI_max:.2%}\nObtido com um ccp de: {ccp_max}')

# árvore ótima
clf = DecisionTreeClassifier(criterion='gini',
                                max_depth = 30, 
                                random_state=42,
                                ccp_alpha=ccp_max).fit(X_train, y_train)

# plota a árvore
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns.tolist(), class_names=['Adimplente', 'Inadimplente'], filled=True)
plt.show()

# confusion_matrix faz a comparação por uma tabela cruzada
cm = confusion_matrix(y, clf.predict(X))

# accuracy_score calcula o percentual de acertos
ac = accuracy_score(y, clf.predict(X))

# pondera para forçar a distribuição da target como uniforme
bac = balanced_accuracy_score(y, clf.predict(X))

# mostra os resultados
print(f'\nA acurácia da árvore é: {ac:.1%}')
print(f'A acurácia balanceada da árvore é: {bac:.1%}')

# visualização gráfica
sns.heatmap(cm, 
            annot=True, fmt='d', cmap='viridis', 
            xticklabels=['Adimplente', 'Inadimplente'], 
            yticklabels=['Adimplente', 'Inadimplente'])
plt.show()

# relatório de classificação do scikit
print('\n', classification_report(y, clf.predict(X)))