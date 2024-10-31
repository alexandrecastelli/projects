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

    dados = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='Data')
    pistas = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='Tracks')
    configurações = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='Settings')
    textos = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='Text')
    pr = pd.read_excel('/workspaces/projects/ml1/ml1.xlsx', sheet_name='PR')

    # renomeia as colunas
    dados = dados.rename(columns={'Date': 'data', 
                                'Version': 'versão', 
                                'Performance': 'desempenho',
                                'Season': 'temporada',
                                'Round': 'etapa',
                                'Grand Prix': 'grande prêmio',
                                'Qualifying': 'classificação',
                                'Sprint': 'sprint',
                                'Race': 'principal',
                                'Driver': 'piloto',
                                'Team': 'equipe',
                                'Best Lap': 'melhor volta'})

    dados['data'] = pd.to_datetime(dados['data']).dt.strftime('%d-%m-%Y')
    # dados['temporada'] = dados['temporada'].astype(str)

    dados = dados.applymap(lambda x: x.lower() if isinstance(x, str) else x)

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

    dados['principal pr'] = np.select([dados['principal'] == 1,
                                    dados['principal'] == 2,                                    
                                    dados['principal'] == 3,
                                    dados['principal'] == 4,
                                    dados['principal'] == 5,
                                    dados['principal'] == 6,
                                    dados['principal'] == 7,
                                    dados['principal'] == 8,
                                    dados['principal'] == 9,
                                    dados['principal'] == 10,
                                    dados['principal'] == 11,
                                    dados['principal'] == 12,
                                    dados['principal'] == 13,
                                    dados['principal'] == 14,
                                    dados['principal'] == 15,
                                    dados['principal'] == 16,
                                    dados['principal'] == 17,
                                    dados['principal'] == 18,
                                    dados['principal'] == 19,
                                    dados['principal'] == 20],
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

    dados['sprint pr'] = np.select([dados['sprint'] == 1,
                                    dados['sprint'] == 2,
                                    dados['sprint'] == 3,
                                    dados['sprint'] == 4,
                                    dados['sprint'] == 5,
                                    dados['sprint'] == 6,
                                    dados['sprint'] == 7,
                                    dados['sprint'] == 8,
                                    dados['sprint'] == 9,
                                    dados['sprint'] == 10,
                                    dados['sprint'] == 11,
                                    dados['sprint'] == 12,
                                    dados['sprint'] == 13,
                                    dados['sprint'] == 14,
                                    dados['sprint'] == 15,
                                    dados['sprint'] == 16,
                                    dados['sprint'] == 17,
                                    dados['sprint'] == 18,
                                    dados['sprint'] == 19,
                                    dados['sprint'] == 20],
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

    dados['classificação pr'] = np.select([dados['classificação'] == 1,
                                        dados['classificação'] == 2,
                                        dados['classificação'] == 3,
                                        dados['classificação'] == 4,
                                        dados['classificação'] == 5,
                                        dados['classificação'] == 6,
                                        dados['classificação'] == 7,
                                        dados['classificação'] == 8,
                                        dados['classificação'] == 9,
                                        dados['classificação'] == 10,
                                        dados['classificação'] == 11,
                                        dados['classificação'] == 12,
                                        dados['classificação'] == 13,
                                        dados['classificação'] == 14,
                                        dados['classificação'] == 15,
                                        dados['classificação'] == 16,
                                        dados['classificação'] == 17,
                                        dados['classificação'] == 18,
                                        dados['classificação'] == 19,
                                        dados['classificação'] == 20],
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

    dados['pr'] = dados['principal pr'] + dados['sprint pr'] + dados['classificação pr']   

    dados.drop(columns=['principal pr', 'sprint pr', 'classificação pr'], inplace=True)

    return dados, pistas, configurações, textos, pr

dados, pistas, configurações, textos, pr = iniciar()

# de acordo com a preferência do usuário

col1, col2, col3 = st.columns(3)

with col1:
    temporada = st.selectbox(
        'temporada',
        np.append(sorted(dados['temporada'].unique(), reverse=True), 'todas')
        )

with col2:
    tabela = st.selectbox(
        'tabela',
        ['etapas','pilotos','equipes','construtores','pr']
        )

with col3:
    auxiliar = st.selectbox(
        'opção',
        ['1','2','3','4']
        )

if temporada == 'todas':
    st.dataframe(dados, hide_index=True, use_container_width=True)
else:
    st.dataframe(dados[dados['temporada'].astype(str) == temporada], hide_index=True, use_container_width=True)