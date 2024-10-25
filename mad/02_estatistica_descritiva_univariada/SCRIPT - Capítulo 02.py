##############################################################################
#                     MANUAL DE ANÁLISE DE DADOS                             #
#                Luiz Paulo Fávero e Patrícia Belfiore                       #
#                            Capítulo 02                                     #
##############################################################################
#!/usr/bin/env python
# coding: utf-8

##############################################################################
#                     IMPORTAÇÃO DOS PACOTES NECESSÁRIOS                     #
##############################################################################

import pandas as pd #manipulação de dados em formato de dataframe
import numpy as np #biblioteca para operações matemáticas multidimensionais
from scipy.stats import skew #cálculo da assimetria
from scipy.stats import kurtosis #cálculo da curtose
import seaborn as sns #biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt #biblioteca de visualização de dados
import stemgraphic #biblioteca para elaboração do gráfico de ramo-e-folhas
from scipy.stats import norm #para plotagem da distribuição normal no histograma


#%%
##############################################################################
#               DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Cotações'                 #
##############################################################################

# Carregamento da base de dados 'Cotações'
df_cotacoes = pd.read_csv('Cotações.csv', delimiter=',')

# Visualização da base de dados 'Cotações'
df_cotacoes

# Tabela de frequências absolutas (contagem) e relativas (%) da variável 'preco'
contagem = df_cotacoes['preco'].value_counts()
percent = df_cotacoes['preco'].value_counts(normalize=True)
pd.concat([contagem, percent], axis=1, keys=['contagem', '%'], sort=True)

# Estatísticas descritivas univariadas da variável 'preco'
df_cotacoes['preco'].describe()

#%%
# Medidas de assimetria e curtose para a variável 'preco'
skew(df_cotacoes, axis=0, bias=True) # igual ao Stata
skew(df_cotacoes, axis=0, bias=False) # igual ao SPSS
kurtosis(df_cotacoes, axis=0, bias=False) # igual ao SPSS

#%%
##############################################################################
#   GRÁFICOS: HISTOGRAMA, RAMO-E-FOLHAS E BOXPLOT PARA A VARIÁVEL 'preco'    #
##############################################################################

# Histograma
plt.figure(figsize=(15,10))
sns.histplot(data=df_cotacoes, x='preco', bins=7, color='darkorchid')
plt.xlabel('Preço', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.show()

#%%
# Histograma com Kernel density estimation (KDE)
plt.figure(figsize=(15,10))
sns.histplot(data=df_cotacoes, x='preco', kde=True, bins=7, color='darkorchid')
plt.xlabel('Preço', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.legend(['Kernel density estimation (KDE)'], fontsize=17)
plt.show()

#%%
# Histograma com curva normal
plt.figure(figsize=(15,10))
mu, std = norm.fit(df_cotacoes['preco'])
plt.hist(df_cotacoes['preco'], bins=7, density=True, alpha=0.6, color='silver')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, color='darkorchid')
plt.xlabel('Preço', fontsize=20)
plt.ylabel('Densidade', fontsize=20)
plt.legend(['Curva Normal'], fontsize=17)
plt.show()

#%%
# Gráfico de ramo-e-folhas para a variável 'preco'
stemgraphic.stem_graphic(df_cotacoes['preco'], scale = 0)

#%%
# Boxplot da variável 'preco' - pacote 'matplotlib'
plt.figure(figsize=(15,10))
plt.boxplot(df_cotacoes['preco'])
plt.title('Preço', fontsize=17)
plt.ylabel('Preço', fontsize=16)
plt.show()

#%%
# Boxplot da variável 'preco' - pacote 'seaborn'
plt.figure(figsize=(15,10))
sns.boxplot(df_cotacoes['preco'], linewidth=2, orient='v', color='purple')
sns.stripplot(df_cotacoes['preco'], color="orange", jitter=0.1, size=7)
plt.title('Preço', fontsize=17)
plt.xlabel('Preço', fontsize=16)
plt.show()


##############################################################################