##############################################################################
#                     MANUAL DE ANÁLISE DE DADOS                             #
#                Luiz Paulo Fávero e Patrícia Belfiore                       #
#                            Capítulo 08                                     #
##############################################################################
#!/usr/bin/env python
# coding: utf-8

##############################################################################
#                     IMPORTAÇÃO DOS PACOTES NECESSÁRIOS                     #
##############################################################################

import pandas as pd #manipulação de dados em formato de dataframe
from scipy import stats #testes estatísticos
import numpy as np #biblioteca para operações matemáticas multidimensionais
from scipy.stats import chisquare #teste qui-quadrado
import statsmodels.stats.descriptivestats as smsd #teste dos sinais
from statsmodels.stats.contingency_tables import mcnemar #teste de McNemar
from scipy.stats import wilcoxon #teste de Wilcoxon
from scipy.stats import mannwhitneyu #teste de Mann-Whitney
from mlxtend.evaluate import cochrans_q #teste de Cochran
from scipy.stats import kruskal #teste de Kruskal-Wallis


##############################################################################
#             DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Teste_Binomial'             #
##############################################################################

# Carregamento da base de dados 'Teste_Binomial'
df_binomial = pd.read_csv('Teste_Binomial.csv', delimiter=',')

# Visualização da base de dados 'Teste_Binomial'
df_binomial

# Determinação do número de valores iguais a '1' na variável 'metodo'
freq = pd.Series(df_binomial['metodo']).value_counts()
k = freq.get(1)

# Dimensão da variável 'metodo'
n = df_binomial.shape[0]

# Teste binomial para verificação se os dois grupos não apresentam proporções
#estatisticamente diferentes (p = q = 0.5), ao nível de confiança de 95%
# Teste bilateral (argumento 'two-sided')
p = stats.binom_test(k, n, p=0.5, alternative='two-sided')

# Interpretação
print('p-value=%.4f' % (p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')


##############################################################################
#        DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Qui_Quadrado_Uma_Amostra'        #
##############################################################################

# Carregamento da base de dados 'Qui_Quadrado_Uma_Amostra'
df_qui1 = pd.read_csv('Qui_Quadrado_Uma_Amostra.csv', delimiter=',')

# Visualização da base de dados 'Qui_Quadrado_Uma_Amostra'
df_qui1

# Tabela de frequências para a variável 'dia_da_semana'
freq = df_qui1['dia_da_semana'].value_counts(sort=False)

# Teste qui-quadrado para uma amostra
stat, p = chisquare(freq)

# Interpretação
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')


##############################################################################
#        DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Teste_Sinais_Uma_Amostra'        #
##############################################################################

# Carregamento da base de dados 'Teste_Sinais_Uma_Amostra'
df_sinais1 = pd.read_csv('Teste_Sinais_Uma_Amostra.csv', delimiter=',')

# Visualização da base de dados 'Teste_Sinais_Uma_Amostra'
df_sinais1

# Teste dos sinais para uma amostra (comparação com a idade = 65)
stat, p = smsd.sign_test(df_sinais1['idade'], 65)

# Interpretação
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')


##############################################################################
#              DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Teste_McNemar'             #
##############################################################################

# Carregamento da base de dados 'Teste_McNemar'
df_mcnemar = pd.read_csv('Teste_McNemar.csv', delimiter=',')

# Visualização da base de dados 'Teste_McNemar'
df_mcnemar

# Tabela de frequências observadas para as variáveis 'antes' e 'depois'
crosstab = pd.crosstab(df_mcnemar['antes'],
                       df_mcnemar['depois'],
                       margins = False)

# Teste de McNemar
print(mcnemar(crosstab, exact=False))


##############################################################################
# DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Teste_Sinais_Duas_Amostras_Emparelhadas'#
##############################################################################

# Carregamento da base de dados 'Teste_Sinais_Duas_Amostras_Emparelhadas'
df_sinais2 = pd.read_csv('Teste_Sinais_Duas_Amostras_Emparelhadas.csv',
                         delimiter=',')

# Visualização da base de dados 'Teste_Sinais_Duas_Amostras_Emparelhadas'
df_sinais2

# Teste dos sinais para duas amostras emparelhadas
diferenca = df_sinais2['antes'] - df_sinais2['depois']
stat, p = smsd.sign_test(diferenca, 0)

# Interpretação
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')


##############################################################################
#            DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Teste_Wilcoxon'              #
##############################################################################

# Carregamento da base de dados 'Teste_Wilcoxon'
df_wilcoxon = pd.read_csv('Teste_Wilcoxon.csv', delimiter=',')

# Visualização da base de dados 'Teste_Wilcoxon'
df_wilcoxon

# Teste de Wilcoxon
stat, p = wilcoxon(df_wilcoxon['antes'],
                   df_wilcoxon['depois'],
                   alternative='two-sided')

# Interpretação
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')


##############################################################################
#              DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'PlanoSaude'                #
##############################################################################

# Carregamento da base de dados 'PlanoSaude'
df_planosaude = pd.read_csv('PlanoSaude.csv', delimiter=',')

# Visualização da base de dados 'PlanoSaude'
df_planosaude

# Tabela de frequências observadas para as variáveis 'operadora' e 'satisfacao'
crosstab = pd.crosstab(df_planosaude['operadora'],
                       df_planosaude['satisfacao'],
                       margins = False)
crosstab

# Teste qui-quadrado para duas amostras independentes
stat, p, df, freq = stats.chi2_contingency(crosstab)

# Interpretação
print('Statistics=%.4f, df=%.1g, p-value=%.4f' % (stat, df, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')


##############################################################################
#           DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Teste_Mann_Whitney'           #
##############################################################################

# Carregamento da base de dados 'Teste_Mann_Whitney'
df_mwhitney = pd.read_csv('Teste_Mann_Whitney.csv', delimiter=',')

# Visualização da base de dados 'Teste_Mann_Whitney'
df_mwhitney

# Teste de Mann-Whitney
array = np.array(df_mwhitney['diametro'])
array1 = array[df_mwhitney['maquina'] == 'a']
array2 = array[df_mwhitney['maquina'] == 'b']

stat, p = mannwhitneyu(array1, array2, alternative="two-sided", method="auto")

# Interpretação
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')


##############################################################################
#            DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Teste_Q_Cochran'             #
##############################################################################

# Carregamento da base de dados 'Teste_Q_Cochran'
df_cochran = pd.read_csv('Teste_Q_Cochran.csv', delimiter=',')

# Visualização da base de dados 'Teste_Q_Cochran'
df_cochran

# Teste Q de Cochran
df_cochran['ytrue'] = 0
ytrue = np.array(df_cochran['ytrue'])

df_cochran['ymodel1'] = np.where(
    df_cochran['a'] == 'satisfeito', 1, 0) 
ymodel1 = np.array(df_cochran['ymodel1'])

df_cochran['ymodel2'] = np.where(
    df_cochran['b'] == 'satisfeito', 1, 0) 
ymodel2 = np.array(df_cochran['ymodel2'])

df_cochran['ymodel3'] = np.where(
    df_cochran['c'] == 'satisfeito', 1, 0) 
ymodel3 = np.array(df_cochran['ymodel3'])

stat, p = cochrans_q(ytrue, ymodel1, ymodel2, ymodel3)

# Interpretação
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')


##############################################################################
#             DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Teste_Friedman'             #
##############################################################################

# Carregamento da base de dados 'Teste_Friedman'
df_friedman = pd.read_csv('Teste_Friedman.csv', delimiter=',')

# Visualização da base de dados 'Teste_Friedman'
df_friedman

# Teste de Friedman
stat, p = stats.friedmanchisquare(df_friedman['at'],
                                  df_friedman['pt'],
                                  df_friedman['d3m'])

# Interpretação
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')


##############################################################################
#  DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Qui_Quadrado_k_Amostras_Independentes' #
##############################################################################

# Carregamento da base de dados 'Qui_Quadrado_k_Amostras_Independentes'
df_quik = pd.read_csv('Qui_Quadrado_k_Amostras_Independentes.csv',
                          delimiter=',')

# Visualização da base de dados 'Qui_Quadrado_k_Amostras_Independentes'
df_quik

# Tabela de frequências observadas para as variáveis 'produtividade' e 'turno'
crosstab = pd.crosstab(df_quik['produtividade'],
                       df_quik['turno'],
                       margins = False)
crosstab

# Teste qui-quadrado para k amostras independentes
stat, p, df, freq = stats.chi2_contingency(crosstab)

# Interpretação
print('Statistics=%.4f, df=%.1g, p-value=%.4f' % (stat, df, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')


##############################################################################
#          DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Teste_Kruskal_Wallis'          #
##############################################################################

# Carregamento da base de dados 'Teste_Kruskal_Wallis'
df_kwallis = pd.read_csv('Teste_Kruskal_Wallis.csv', delimiter=',')

# Visualização da base de dados 'Teste_Kruskal_Wallis'
df_kwallis

# Teste de Kruskal-Wallis
array = np.array(df_kwallis['resultado'])
array1 = array[df_kwallis['tratamento'] == 1]
array2 = array[df_kwallis['tratamento'] == 2]
array3 = array[df_kwallis['tratamento'] == 3]

stat, p = kruskal(array1, array2, array3)

# Interpretação
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')


##############################################################################