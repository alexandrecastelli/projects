
# carrega as bibliotecas necessárias
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score

# seleciona as features e a target
X = dados.drop(columns = ['inadimplencia'])
y = dados['inadimplencia']

# define as bases de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

# confere o número de linhas e colunas
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# cria o modelo de árvore de decisão
arvore = DecisionTreeClassifier(criterion='gini', 
                                max_depth=30, 
                                random_state=42,
                                ccp_alpha=0)

# treina o modelo
arvore.fit(X_train, y_train)

# cost-complexity pruning path
ccp_path = pd.DataFrame(arvore.cost_complexity_pruning_path(X_train, y_train))

# tunando a árvore
GINIs = []

for ccp in ccp_path['ccp_alphas']:
    arvore = DecisionTreeClassifier(criterion='gini', max_depth=30, random_state=42, ccp_alpha=ccp)

    # treina o modelo
    arvore.fit(X_train, y_train)
    AUC = roc_auc_score(y_test, arvore.predict_proba(X_test)[:, -1])
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

# arvore ótima
arvore = DecisionTreeClassifier(criterion='gini',
                                max_depth = 30, 
                                random_state=42,
                                ccp_alpha=ccp_max).fit(X_train, y_train)

# plota a árvore
plt.figure(figsize=(20, 10))
plot_tree(arvore, feature_names=X.columns.tolist(), class_names=['Adimplente', 'Inadimplente'], filled=True)
plt.show()

# confusion_matrix faz a comparação por uma tabela cruzada
cm = confusion_matrix(y, arvore.predict(X))

# accuracy_score calcula o percentual de acertos
ac = accuracy_score(y, arvore.predict(X))

# pondera para forçar a distribuição da target como uniforme
bac = balanced_accuracy_score(y, arvore.predict(X))

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
print('\n', classification_report(y, arvore.predict(X)))