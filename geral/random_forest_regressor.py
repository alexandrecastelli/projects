#%%
# carrega os pacotes

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from functions import descriptive

#%%
# carrega os dados

tips = sns.load_dataset('tips')

#%%
# cria a coluna de percentual de gorjeta

tips['pct_tip'] = tips['tip'] / (tips['total_bill'] - tips['tip'])

#%%
#  explora os dados

sns.boxplot(x='pct_tip', data=tips)
plt.show()

#%%
# remove outliers (opcional)

tips = tips[tips['pct_tip'] < 1]

#%%
# prepara os dados para o modelo

X = tips[['sex', 'smoker', 'day', 'time', 'size', 'total_bill']]
y = tips['pct_tip']

#%% 
# análises descritivas

for col in tips.columns[:-1]:
    descriptive(df=tips, target='pct_tip', features=col)

#%% 
# codifica as variáveis categóricas (se necessário)

X = pd.get_dummies(X)

#%% 
# divide os dados em treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% 
# cria o modelo de árvore de decisão

model = RandomForestRegressor(random_state=42)

#%% 
# treina o modelo

model.fit(X_train, y_train)

#%% 
# calcula as previsões

y_pred = model.predict(X_test)

#%% 
# avalia o modelo (usando MSE como exemplo)

mse = mean_squared_error(y_test, y_pred)
rquad = r2_score(y_test,y_pred)
print(f'Mean Squared Error: {mse:.5f} | R-quadrado={rquad:,.1f}')

#%% 
# define os parâmetros para o grid search 

params = { 'n_estimators': [100], 
          'max_depth': [2, 3, 6], 
          'min_samples_split': [2, 5],
          'max_features': [2, 5]
          }

grid = GridSearchCV(RandomForestRegressor(), 
                    params, 
                    cv=5)

grid.fit(X_train, y_train)
print(grid.best_params_)

#%% 
# avalia o modelo na base de teste

y_pred = grid.best_estimator_.predict(X_test)

r2treino = r2_score(y_train,y_train)
r2teste = r2_score(y_test,y_pred)

print(f"R-quadrado na base de teste: {r2teste:,.2%}")

#%% 
# visualiza o gráfico do resultado

sns.pointplot(x=pd.qcut(y_pred, 5, duplicates='drop'), y=y_test)
plt.show()