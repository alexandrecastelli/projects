# python -m venv .venv
# source .venv/bin/activate
# deactivate
# pip freeze > requirements.txt
# pip install -r requirements.txt
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import gdown
from datetime import datetime

st.set_page_config(
    page_title='Master League 1',
    page_icon='/workspaces/projects/ml1/images/ml1.png',
    layout='wide',
    initial_sidebar_state='collapsed',
    menu_items={'Get Help': 'mailto:john@example.com',
                'Report a bug': 'mailto:john@example.com',
                'About': 'text'}
)

@st.cache_data
def iniciar():
    # gdown.download('https://docs.google.com/spreadsheets/d/1anYsLCtlv3PCFzfaq2E_qOMKSzR1biQy/export?format=xlsx', 'ml1/ml1.xlsx')

    calendarios = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='calendarios')
    circuitos = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='circuitos')
    configuracoes = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='configuracoes')
    pr = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='pr')
    resultados = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='resultados')
    resumos = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='resumos')

    calendarios['data'] = pd.to_datetime(calendarios['data']).dt.strftime('%d-%m-%Y')

    # resultados = resultados.map(lambda x: x.upper() if isinstance(x, str) else x)

    resultados = calendarios.merge(resultados, on=['temporada','etapa'], how='left').drop(columns=['data','versao','desempenho'])

    # mapeamentos
    sprint_pontos_map = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
    principal_pontos_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    classification_pr_map = {i: 21 - i for i in range(1, 21)}

    # usando replace (vetorizado)
    resultados['sprint_pontos'] = resultados['sprint'].replace(sprint_pontos_map).fillna(0).astype(int)
    resultados['principal_pontos'] = resultados['principal'].replace(principal_pontos_map).fillna(0).astype(int)
    resultados['classificacao_pr'] = resultados['classificacao'].replace(classification_pr_map).fillna(0).astype(int)

    # usando map
    # resultados['sprint_pontos'] = resultados['sprint'].map(sprint_pontos_map).fillna(0).astype(int)
    # resultados['principal_pontos'] = resultados['principal'].map(principal_pontos_map).fillna(0).astype(int)
    # resultados['classificacao_pr'] = resultados['classificacao'].map(classification_pr_map).fillna(0).astype(int)

    resultados['sprint_pr'] = (21 - resultados['classificacao_pr']) * 3
    resultados['principal_pr'] = (21 - resultados['classificacao_pr']) * 9

    resultados['pontos'] = resultados['principal_pontos'] + resultados['sprint_pontos']    
    resultados.drop(columns=['principal_pontos', 'sprint_pontos'], inplace=True)

    resultados['pr'] = resultados['principal_pr'] + resultados['sprint_pr'] + resultados['classificacao_pr']   
    resultados.drop(columns=['principal_pr', 'sprint_pr', 'classificacao_pr'], inplace=True)

    return calendarios, circuitos, configuracoes, pr, resultados, resumos

calendarios, circuitos, configuracoes, pr, resultados, resumos = iniciar()

# de acordo com a preferência do usuário

col1, col2, col3 = st.columns(3)

with col1:
    temporada_selecionada = st.selectbox(
        '',
        resultados['temporada'].unique()
        )

with col2:
    tabela_selecionada = st.selectbox(
        '',
        ['Corridas', 'Pilotos', 'Equipes', 'Power Ranking']
        )

with col3:
    opcao_selecionada = st.selectbox(
        '',
        resultados.query('temporada == @temporada_selecionada')['corrida'].unique() if tabela_selecionada == 'Corridas'
        else resultados.query('temporada == @temporada_selecionada')['piloto'].unique() if tabela_selecionada == 'Pilotos' or tabela_selecionada == 'Power Ranking'
        else resultados.query('temporada == @temporada_selecionada')['equipe'].unique()
        )

# resultados['piloto'] = resultados['piloto'] + '🥇'

if tabela_selecionada == 'Corridas':
    st.dataframe(resultados.query('temporada == @temporada_selecionada'), hide_index=True, use_container_width=True)
elif tabela_selecionada == 'Pilotos':
    st.dataframe(resultados.query('temporada == @temporada_selecionada'), hide_index=True, use_container_width=True)
elif tabela_selecionada == 'Equipes':
    st.dataframe(resultados.query('temporada == @temporada_selecionada'), hide_index=True, use_container_width=True)
elif tabela_selecionada == 'Power Ranking':
    st.dataframe(resultados.query('temporada == @temporada_selecionada'), hide_index=True, use_container_width=True)