##############################################################################
#                     MANUAL DE ANÁLISE DE DADOS                             #
#                Luiz Paulo Fávero e Patrícia Belfiore                       #
#                            Capítulo 03                                     #
##############################################################################
#!/usr/bin/env python
# coding: utf-8

##############################################################################
#                     IMPORTAÇÃO DOS PACOTES NECESSÁRIOS                     #
##############################################################################

import pandas as pd #manipulação de dados em formato de dataframe
from scipy.stats import chi2_contingency #estatística qui-quadrado
from scipy.stats.contingency import association #medidas de associação
import matplotlib.pyplot as plt #biblioteca de visualização de dados
import seaborn as sns #biblioteca de visualização de informações estatísticas
import numpy as np #biblioteca para operações matemáticas multidimensionais
from scipy.stats import pearsonr #cálculo da correlação de Pearson


##############################################################################
#             DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'PlanoSaude'                 #
##############################################################################

# Carregamento da base de dados 'PlanoSaude'
df_planosaude = pd.read_csv('PlanoSaude.csv', delimiter=',')

# Visualização da base de dados 'PlanoSaude'
df_planosaude

# Tabela de contingência para as variáveis 'operadora' e 'satisfacao'
# Tabela completa com soma nas linhas e colunas
tabela_completa = pd.crosstab(df_planosaude['operadora'],
                              df_planosaude['satisfacao'], margins=True)
tabela_completa

# Tabela de contingência propriamente dita
tabela_conting = pd.crosstab(df_planosaude['operadora'],
                             df_planosaude['satisfacao'])
tabela_conting

# Medida de associação - estatística qui-quadrado e teste
chi2, pvalor, df, freq_esp = chi2_contingency(tabela_conting)
pd.DataFrame({'Qui-quadrado':[chi2],
              'Graus de liberdade':[df],
              'p-value':[pvalor]})

# Resíduos da tabela de contingência
residuos = tabela_conting - freq_esp
residuos


##############################################################################
#          DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Segmentação_Mercado'           #
##############################################################################

# Carregamento da base de dados 'Segmentação_Mercado'
df_segmentacao = pd.read_csv('Segmentação_Mercado.csv', delimiter=',')

# Visualização da base de dados 'Segmentação_Mercado'
df_segmentacao

# Tabela de contingência para as variáveis 'roupa' e 'regiao'
tabela_conting = pd.crosstab(df_segmentacao['roupa'],
                             df_segmentacao['regiao'])
tabela_conting

# Medidas de associação
# Coeficiente de contingência
association(tabela_conting, method='pearson')

# Coeficiente V de Cramer
association(tabela_conting, method='cramer')


##############################################################################
#                DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Notas'                   #
##############################################################################

# Carregamento da base de dados 'Notas'
df_notas = pd.read_csv('Notas.csv', delimiter=',')

# Visualização da base de dados 'Notas'
df_notas

# Coeficiente de correlação de Spearman
df_notas.corr(method="spearman")


##############################################################################
#             DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Renda_Estudo'               #
##############################################################################

# Carregamento da base de dados 'Renda_Estudo'
df_renda = pd.read_csv('Renda_Estudo.csv', delimiter=',')

# Visualização da base de dados 'Renda_Estudo'
df_renda

# Diagrama de dispersão
plt.figure(figsize=(15,10))
sns.scatterplot(data=df_renda, x='anosdeestudo', y='rendafamiliar')
plt.xlabel('Anos de Estudo', fontsize=16)
plt.ylabel('Renda Familiar', fontsize=16)
plt.show

# Medidas de correlação
# Covariância
np.cov(df_renda['rendafamiliar'], df_renda['anosdeestudo'])

# Coeficiente de correlação de Pearson
# Maneira 1
df_renda.corr(method="pearson")

# Maneira 2 (pacote numpy)
np.corrcoef(df_renda['rendafamiliar'], df_renda['anosdeestudo'])

# Maneira 3 (pacote scipy.stats)
pearsonr(df_renda['rendafamiliar'], df_renda['anosdeestudo'])

# Mapa de calor com correlação entre as variáveis
plt.figure(figsize=(15,10))
sns.heatmap(df_renda.corr(), annot=True, cmap = plt.cm.viridis,
            annot_kws={'size':20})
plt.show

# Gráfico com diagramas de dispersão e histogramas das variáveis
plt.figure(figsize=(15,10))
sns.pairplot(df_renda)
plt.show

# Gráfico com diagramas de dispersão, histogramas das variáveis e
#correlações de Pearson

# Definição da função 'corrfunc'
def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.4, .9), xycoords=ax.transAxes)

plt.figure(figsize=(15,10))
graph = sns.pairplot(df_renda)
graph.map(corrfunc)
plt.show()

##############################################################################