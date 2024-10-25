##############################################################################
#                     MANUAL DE ANÁLISE DE DADOS                             #
#                Luiz Paulo Fávero e Patrícia Belfiore                       #
#                            Capítulo 07                                     #
##############################################################################
#!/usr/bin/env python
# coding: utf-8

##############################################################################
#                     IMPORTAÇÃO DOS PACOTES NECESSÁRIOS                     #
##############################################################################

import pandas as pd #manipulação de dados em formato de dataframe
from scipy import stats #testes estatísticos
from statstests.tests import shapiro_francia #teste de Shapiro-Francia
import seaborn as sns #biblioteca de visualização de informações estatísticas
import numpy as np #biblioteca para operações matemáticas multidimensionais
import matplotlib.pyplot as plt #biblioteca de visualização de dados
from scipy.stats import f_oneway #biblioteca para one-way ANOVA
import statsmodels.api as sm #estimação de modelo para two-way ANOVA
from statsmodels.formula.api import ols #estimação de modelo para two-way ANOVA


#%%
##############################################################################
#       DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Produção_MáquinasAgrícolas'       #
##############################################################################

# Carregamento da base de dados 'Produção_MáquinasAgrícolas'
df_maquinas = pd.read_csv('Produção_MáquinasAgrícolas.csv', delimiter=',')

# Visualização da base de dados 'Produção_MáquinasAgrícolas'
df_maquinas

# Teste de Kolmogorov-Smirnov
stat, p = stats.kstest((df_maquinas['produção']
                        -df_maquinas['produção'].mean())/df_maquinas['produção'].std(),
                       "norm", alternative='two-sided')

# Interpretação
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')

#%%
##############################################################################
#            DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Produção_Aviões'             #
##############################################################################

# Carregamento da base de dados 'Produção_Aviões'
df_avioes = pd.read_csv('Produção_Aviões.csv', delimiter=',')

# Visualização da base de dados 'Produção_Aviões'
df_avioes

# Teste de Shapiro-Wilk
stat, p = stats.shapiro(df_avioes['produção'])

# Interpretação
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')

#%%
##############################################################################
#          DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Produção_Bicicletas'           #
##############################################################################

# Carregamento da base de dados 'Produção_Bicicletas'
df_bicicletas = pd.read_csv('Produção_Bicicletas.csv', delimiter=',')

# Visualização da base de dados 'Produção_Bicicletas'
df_bicicletas

# Teste de Shapiro-Francia
# Instalação e carregamento da função 'shapiro_francia' do pacote
#'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/
# pip install statstests
shapiro_francia(df_bicicletas['produção'])

# Interpretação
teste_sf = shapiro_francia(df_bicicletas['produção']) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

#%%
##############################################################################
#            DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Atendimentos_Loja'           #
##############################################################################

# Carregamento da base de dados 'Atendimentos_Loja'
df_atendimentos = pd.read_csv('Atendimentos_Loja.csv', delimiter=',')

# Visualização da base de dados 'Atendimentos_Loja'
df_atendimentos

# Teste de Levene para homogeneidade de variâncias
array = np.array(df_atendimentos['atendimentos'])
array1 = array[df_atendimentos['loja'] == 1]
array2 = array[df_atendimentos['loja'] == 2]
array3 = array[df_atendimentos['loja'] == 3]

stat, p = stats.levene(array1, array2, array3, center='mean')

# Interpretação
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')

#%%
##############################################################################
#            DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Exemplo9_Teste_t'            #
##############################################################################

# Carregamento da base de dados 'Exemplo9_Teste_t'
df_exemplo9 = pd.read_csv('Exemplo9_Teste_t.csv', delimiter=',')

# Visualização da base de dados 'Exemplo9_Teste_t'
df_exemplo9

# Teste t de Student para uma amostra
# Especifica-se mu = 18 (valor a ser testado)
stat, p = stats.ttest_1samp(df_exemplo9['tempo'], popmean=18)

# Interpretação
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')

#%%
##############################################################################
#   DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Teste_t_Duas_Amostras_Independentes'  #
##############################################################################

# Carregamento da base de dados 'Teste_t_Duas_Amostras_Independentes'
df_testet_ind = pd.read_csv('Teste_t_Duas_Amostras_Independentes.csv',
                            delimiter=',')

# Visualização da base de dados 'Teste_t_Duas_Amostras_Independentes'
df_testet_ind

# Teste t de Student para Duas Amostras Independentes
array = np.array(df_testet_ind['tempo'])
array1 = array[df_testet_ind['fornecedor'] == 1]
array2 = array[df_testet_ind['fornecedor'] == 2]

stat, p = stats.ttest_ind(array1, array2, equal_var=True)

# Interpretação
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')

#%%
##############################################################################
#   DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'Teste_t_Duas_Amostras_Emparelhadas'   #
##############################################################################

# Carregamento da base de dados 'Teste_t_Duas_Amostras_Emparelhadas'
df_testet_emp = pd.read_csv('Teste_t_Duas_Amostras_Emparelhadas.csv',
                            delimiter=',')

# Visualização da base de dados 'Teste_t_Duas_Amostras_Emparelhadas'
df_testet_emp

# Teste t de Student para Duas Amostras Emparelhadas
stat, p = stats.ttest_rel(df_testet_emp['antes'], df_testet_emp['depois'])

# Interpretação
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')

#%%
##############################################################################
#            DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'ANOVA_Um_Fator'              #
##############################################################################

# Carregamento da base de dados 'ANOVA_Um_Fator'
df_anova1 = pd.read_csv('ANOVA_Um_Fator.csv', delimiter=',')

# Visualização da base de dados 'ANOVA_Um_Fator'
df_anova1

# Boxplots da variável 'sacarose' por 'fornecedor' - pacote matplotlib
array = np.array(df_anova1['sacarose'])
array1 = array[df_anova1['fornecedor'] == 1]
array2 = array[df_anova1['fornecedor'] == 2]
array3 = array[df_anova1['fornecedor'] == 3]

data = {'Fornecedor 1': array1,
        'Fornecedor 2': array2,
        'Fornecedor 3': array3}

fig, bp = plt.subplots()
bp.set_title('Boxplots de sacarose por fornecedor')
bp.boxplot(data.values())
bp.set_xticklabels(data.keys())
plt.show()

# Boxplots da variável 'sacarose' por 'fornecedor' - pacote seaborn
plt.figure(figsize=(15,10))
sns.boxplot(data=df_anova1, x='fornecedor', y='sacarose',
            linewidth=2, orient='v', color='darkorchid')
sns.stripplot(data=df_anova1, x='fornecedor', y='sacarose',
              color="orange", jitter=0.1, size=7)
plt.title('Boxplots de sacarose por fornecedor', fontsize=17)
plt.xlabel('Fornecedor', fontsize=16)
plt.ylabel('Sacarose', fontsize=16)
plt.show()

# Teste de Levene para homogeneidade de variâncias
stat, p = stats.levene(array1, array2, array3, center='mean')

# Interpretação do teste de Levene
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')

# ANOVA de um fator
stat, p = f_oneway(array1, array2, array3)

# Interpretação da ANOVA de um fator
print('Statistics=%.4f, p-value=%.4f' % (stat, p))
alpha = 0.05 #nível de significância
if p > alpha:
	print('Não se rejeita H0')
else:
	print('Rejeita-se H0')

#%%
##############################################################################
#          DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'ANOVA_Dois_Fatores'            #
##############################################################################

# Carregamento da base de dados 'ANOVA_Dois_Fatores'
df_anova2 = pd.read_csv('ANOVA_Dois_Fatores.csv', delimiter=',')

# Visualização da base de dados 'ANOVA_Dois_Fatores'
df_anova2

# ANOVA de dois fatores
modelo = ols('tempo ~ C(companhia) + C(dia_da_semana) + C(companhia):C(dia_da_semana)',
             data=df_anova2).fit()
modelo.summary()
sm.stats.anova_lm(modelo, typ=2)

##############################################################################