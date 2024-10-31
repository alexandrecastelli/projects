# python -m venv .venv
# source .venv/bin/activate
# deactivate
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import gdown

st.set_page_config(
    page_title='Master League 1',
    page_icon='/workspaces/projects/ml1/images/ml1.png',
    layout='wide',
    initial_sidebar_state='collapsed',
    menu_items={'Get Help': 'mailto:john@example.com',
                'Report a bug': 'mailto:john@example.com',
                'About': 'text'}
)

# st.logo('/workspaces/projects/ml1/images/ml1.png', size='large')

# gdown.download('https://docs.google.com/spreadsheets/d/1anYsLCtlv3PCFzfaq2E_qOMKSzR1biQy/export?format=xlsx', 'ml1.xlsx')

dados = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='data')
pistas = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='tracks')
configurações = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='settings')
textos = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='text')
pr = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='pr')



# colunas sempre minúsculas
dados.columns = dados.columns.str.lower()

# dados = dados.rename(columns={'A': 'Alpha', 
#                               'B': 'Bravo', 
#                               'C': 'Charlie'})

# dados.columns = ['temporada', 'etapa', 'piloto', 'equipe', 'construtor', 'principal', 'sprint', 'pontos']

# de acordo com a preferência do usuário
dados = dados.applymap(lambda x: x.upper() if isinstance(x, str) else x)

dados['principal pontos'] = np.select([dados['principal'] == 1,
                            dados['principal'] == 2,
                            dados['principal'] == 3,
                            dados['principal'] == 4,
                            dados['principal'] == 5,
                            dados['principal'] == 6,
                            dados['principal'] == 7,
                            dados['principal'] == 8,
                            dados['principal'] == 9,
                            dados['principal'] == 10,
                            dados['principal'] >= 11],
                            [25,
                             18,
                             15,
                             12,
                             10,
                             8,
                             6,
                             4,
                             2,
                             1,
                             0])

dados['sprint pontos'] = np.select([dados['sprint'] == 1,
                            dados['sprint'] == 2,
                            dados['sprint'] == 3,
                            dados['sprint'] == 4,
                            dados['sprint'] == 5,
                            dados['sprint'] == 6,
                            dados['sprint'] == 7,
                            dados['sprint'] == 8,
                            dados['sprint'] >= 9],
                            [8,
                             7,
                             6,
                             5,
                             4,
                             3,
                             2,
                             1,
                             0])

dados['pontos'] = dados['principal pontos'] + dados['sprint pontos']    

dados.drop(columns=['principal pontos', 'sprint pontos'], inplace=True)

col1, col2, col3 = st.columns(3)

with col1:
    temporada = st.selectbox(
        'temporada',
        dados['temporada'].unique()
        )

with col2:
    tabela = st.selectbox(
        'tabela',
        ['etapas','pilotos','equipes','construtores','pr'], 
        )

with col3:
    auxiliar = st.selectbox(
        'opção',
        ['1','2','3','4'], 
        )

st.dataframe(dados[dados['temporada'] == temporada], hide_index=True, use_container_width=True)