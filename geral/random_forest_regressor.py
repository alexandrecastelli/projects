#%% Carrega os pacotes

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from functions import descriptive

#%% Carrega os dados

tips = sns.load_dataset('tips')

#%% Cria a coluna de percentual de gorjeta

tips['pct_tip'] = tips['tip'] / (tips['total_bill'] - tips['tip'])

#%% Explora os dados

sns.boxplot(x='pct_tip', data=tips)
plt.show()

#%% Remove outliers (opcional)

tips = tips[tips['pct_tip'] < 1]

#%% Prepara os dados para o modelo

X = tips[['sex', 'smoker', 'day', 'time', 'size', 'total_bill']]
y = tips['pct_tip']

#%% Analisa estatísticas descritivas

for col in tips.columns[:-1]:
    descriptive(df=tips, target='pct_tip', features=col)

#%% Codifica as variáveis categóricas (se necessário)

X = pd.get_dummies(X)

#%% Treina o modelo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

#%% Avalia o modelo (usando MSE como exemplo)

mse = mean_squared_error(y_test, y_pred)
rquad = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.5f} | R-quadrado={rquad:,.2%}')

#%% Define os parâmetros para o grid search 

param_grid = {'n_estimators': [100],
              'max_depth': [2, 3, 6],
              'min_samples_split': [2, 5],
              'max_features': [2, 5]}

rf_gs = GridSearchCV(RandomForestRegressor(),
                     param_grid,
                     cv=5)

rf_gs.fit(X_train, y_train)
print(rf_gs.best_params_)

#%% Avalia o modelo na base de teste

y_pred = rf_gs.best_estimator_.predict(X_test)

r2treino = r2_score(y_train,y_train)
r2teste = r2_score(y_test, y_pred)

print(f"R-quadrado na base de treino: {r2treino:,.2%}")
print(f"R-quadrado na base de teste: {r2teste:,.2%}")

#%% Visualiza o resultado

sns.pointplot(x=pd.qcut(y_pred, 5, duplicates='drop'), y=y_test)
plt.show()