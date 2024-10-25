##############################################################################
#                     MANUAL DE ANÁLISE DE DADOS                             #
#                Luiz Paulo Fávero e Patrícia Belfiore                       #
#                            Capítulo 01                                     #
##############################################################################
#!/usr/bin/env python
# coding: utf-8

##############################################################################
#                     IMPORTAÇÃO DOS PACOTES NECESSÁRIOS                     #
##############################################################################

import pandas as pd #manipulação de dados em formato de dataframe


##############################################################################
#              DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'VarQuanti'                 #
##############################################################################
# Carregamento da base de dados
df = pd.read_csv('VarQuanti.csv', delimiter=',')

# Visualização da base de dados
df

# Escalas de mensuração das variáveis
df.info()

##############################################################################