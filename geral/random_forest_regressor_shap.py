#%%
# carrega os pacotes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from functions import descriptive, diagnosis
import shap
import matplotlib.pyplot as plt

#%%
# carrega os dados

df = pd.read_csv('houses_to_rent.csv', index_col=0)

df.info()

#%%
# limpa os dados

df['floor'] = df.floor.str.replace('-','NaN').astype('float64')

for var in ['hoa', 'rent amount', 'property tax', 'fire insurance', 'total']:
    df[var] = df[var].str.replace('R$','')\
        .str.replace(',','')\
        .str.replace('Sem info','NaN')\
        .str.replace('Incluso','0').astype('float64')
        
df.info()

X_cols = ['city', 'area', 'rooms', 'bathroom', 'parking spaces', 'floor', 'animal', 'furniture']
y_col = 'total'

X = pd.get_dummies(df[X_cols], drop_first=True)
y = df[y_col]

for col in X_cols:
    descriptive(df, y, col)

#%%
# separa os dados em treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)

#%%
# treina o modelo

rf = RandomForestRegressor()
rf.fit(X, y)

#%%
# avalia o modelo

r2_score(y_test, rf.predict(X_test))

df['pred'] = rf.predict(X)

#%% 
# faz um diagnóstico por variáveis

for col in X_cols:
    diagnosis(df, col, y, 'pred')
    
#%% 
# calcula os 'shap values'

amostra = X_test.sample(frac=0.1)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(amostra)

#%% 
# mostra os gráficos

shap.summary_plot(shap_values, amostra, feature_names=X.columns)

shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                     base_values=explainer.expected_value, 
                                     data=amostra.iloc[0], 
                                     feature_names=amostra.columns))


#%% 
# gera o gráfico manualmente

df_shap = pd.DataFrame(shap_values, columns=X.columns)
df_shap.iloc[0].plot.bar()