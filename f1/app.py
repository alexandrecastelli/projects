import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="ML1 Results",
    layout="wide",
    page_icon="https://www.formula1.com/etc/designs/fom-website/favicon.ico",
    initial_sidebar_state="collapsed",
)

@st.cache_data
def load_data(path):
    calendarios = pd.read_excel(path, sheet_name='calendarios')
    resultados = pd.read_excel(path, sheet_name='resultados')
    # remove first row if it contains season summary text
    if not resultados.iloc[0].apply(lambda x: isinstance(x, (int, float))).all():
        resultados = resultados.drop(index=0).reset_index(drop=True)
    return calendarios, resultados

calendarios, resultados = load_data('../ml1/ml1.xlsx')

st.title('Master League 1 - Resultados')

seasons = sorted(resultados['temporada'].unique())
season = st.selectbox('Temporada', seasons)

calendario_temp = calendarios[calendarios['temporada'] == season]

etapas = calendario_temp['etapa'].tolist()
selected_race = st.selectbox('Etapa', etapas, format_func=lambda x: calendario_temp.loc[calendario_temp['etapa']==x,'corrida'].values[0])

st.subheader('Calendário')
st.table(calendario_temp[['etapa','corrida','data']].reset_index(drop=True))

st.subheader('Resultados da Corrida')
res = resultados[(resultados['temporada']==season) & (resultados['etapa']==selected_race)]
res = res[['classificacao','piloto','equipe','sprint','principal']]
res = res.sort_values('classificacao')
res = res.reset_index(drop=True)
st.table(res)
