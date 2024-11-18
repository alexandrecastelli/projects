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

    resultados['sprint_pontos'] = np.select([resultados['sprint'] == 1,
                                         resultados['sprint'] == 2,
                                         resultados['sprint'] == 3,
                                         resultados['sprint'] == 4,
                                         resultados['sprint'] == 5,
                                         resultados['sprint'] == 6,
                                         resultados['sprint'] == 7,
                                         resultados['sprint'] == 8,
                                         resultados['sprint'] >= 9],
                                         [8,
                                          7,
                                          6,
                                          5,
                                          4,
                                          3,
                                          2,
                                          1,
                                          0])

    resultados['principal_pontos'] = np.select([resultados['principal'] == 1,
                                            resultados['principal'] == 2,
                                            resultados['principal'] == 3,
                                            resultados['principal'] == 4,
                                            resultados['principal'] == 5,
                                            resultados['principal'] == 6,
                                            resultados['principal'] == 7,
                                            resultados['principal'] == 8,
                                            resultados['principal'] == 9,
                                            resultados['principal'] == 10,
                                            resultados['principal'] >= 11],
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

    resultados['pontos'] = resultados['principal_pontos'] + resultados['sprint_pontos']    

    resultados.drop(columns=['principal_pontos', 'sprint_pontos'], inplace=True)

    resultados['classificacao_pr'] = np.select([resultados['classificacao'] == 1,
                                            resultados['classificacao'] == 2,
                                            resultados['classificacao'] == 3,
                                            resultados['classificacao'] == 4,
                                            resultados['classificacao'] == 5,
                                            resultados['classificacao'] == 6,
                                            resultados['classificacao'] == 7,
                                            resultados['classificacao'] == 8,
                                            resultados['classificacao'] == 9,
                                            resultados['classificacao'] == 10,
                                            resultados['classificacao'] == 11,
                                            resultados['classificacao'] == 12,
                                            resultados['classificacao'] == 13,
                                            resultados['classificacao'] == 14,
                                            resultados['classificacao'] == 15,
                                            resultados['classificacao'] == 16,
                                            resultados['classificacao'] == 17,
                                            resultados['classificacao'] == 18,
                                            resultados['classificacao'] == 19,
                                            resultados['classificacao'] == 20],
                                            [20,
                                            19,
                                            18,
                                            17,
                                            16,
                                            15,
                                            14,
                                            13,
                                            12,
                                            11,
                                            10,
                                            9,
                                            8,
                                            7,
                                            6,
                                            5,
                                            4,
                                            3,
                                            2,
                                            1])

    resultados['sprint_pr'] = np.select([resultados['sprint'] == 1,
                                     resultados['sprint'] == 2,
                                     resultados['sprint'] == 3,
                                     resultados['sprint'] == 4,
                                     resultados['sprint'] == 5,
                                     resultados['sprint'] == 6,
                                     resultados['sprint'] == 7,
                                     resultados['sprint'] == 8,
                                     resultados['sprint'] == 9,
                                     resultados['sprint'] == 10,
                                     resultados['sprint'] == 11,
                                     resultados['sprint'] == 12,
                                     resultados['sprint'] == 13,
                                     resultados['sprint'] == 14,
                                     resultados['sprint'] == 15,
                                     resultados['sprint'] == 16,
                                     resultados['sprint'] == 17,
                                     resultados['sprint'] == 18,
                                     resultados['sprint'] == 19,
                                     resultados['sprint'] == 20],
                                     [3*20,
                                     3*19,
                                     3*18,
                                     3*17,
                                     3*16,
                                     3*15,
                                     3*14,
                                     3*13,
                                     3*12,
                                     3*11,
                                     3*10,
                                     3*9,
                                     3*8,
                                     3*7,
                                     3*6,
                                     3*5,
                                     3*4,
                                     3*3,
                                     3*2,
                                     3*1])

    resultados['principal_pr'] = np.select([resultados['principal'] == 1,
                                        resultados['principal'] == 2,                                    
                                        resultados['principal'] == 3,
                                        resultados['principal'] == 4,
                                        resultados['principal'] == 5,
                                        resultados['principal'] == 6,
                                        resultados['principal'] == 7,
                                        resultados['principal'] == 8,
                                        resultados['principal'] == 9,
                                        resultados['principal'] == 10,
                                        resultados['principal'] == 11,
                                        resultados['principal'] == 12,
                                        resultados['principal'] == 13,
                                        resultados['principal'] == 14,
                                        resultados['principal'] == 15,
                                        resultados['principal'] == 16,
                                        resultados['principal'] == 17,
                                        resultados['principal'] == 18,
                                        resultados['principal'] == 19,
                                        resultados['principal'] == 20],
                                        [9*20,
                                         9*19,
                                         9*18,
                                         9*17,
                                         9*16,
                                         9*15,
                                         9*14,
                                         9*13,
                                         9*12,
                                         9*11,
                                         9*10,
                                         9*9,
                                         9*8,
                                         9*7,
                                         9*6,
                                         9*5,
                                         9*4,
                                         9*3,
                                         9*2,
                                         9*1])

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