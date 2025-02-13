#%%  
# funções de ajuda

import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, balanced_accuracy_score, roc_auc_score, roc_curve
    
import seaborn as sns

import matplotlib.pyplot as plt

def descriptive(df, target, features, max_classes=5):

    df = df.copy()
    
    if df[features].nunique() > max_classes:
        df[features] = pd.qcut(df[features], max_classes, duplicates='drop')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.pointplot(data=df, y=target, x=features, ax=ax1)
    
    ax2 = ax1.twinx()
    sns.countplot(data=df, x=features, palette='viridis', alpha=0.5, ax=ax2)
    ax2.set_ylabel('Frequência', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)
    
    plt.show()
    
def evaluate(clf, y, X, rótulos_y=['Not survived', 'Survived'], base='treino'):
    
    pred = clf.predict(X)
    
    y_prob = clf.predict_proba(X)[:, -1]
    
    cm = confusion_matrix(y, pred)
    ac = accuracy_score(y, pred)
    bac = balanced_accuracy_score(y, pred)

    print(f'\nBase de {base}:')
    print(f'A acurácia da árvore é: {ac:.1%}')
    print(f'A acurácia balanceada da árvore é: {bac:.1%}')
    
    auc_score = roc_auc_score(y, y_prob)
    print(f"AUC-ROC: {auc_score:.2%}")
    print(f"GINI: {(2*auc_score-1):.2%}")
    
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='viridis', 
                xticklabels=rótulos_y, 
                yticklabels=rótulos_y)
    
    print('\n', classification_report(y, pred))
    
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title(f"Curva ROC - base de {base}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

def missing(df):
    print(f'Lines: {df.shape[0]} | Columns: {df.shape[1]}')
    return pd.DataFrame({'missing_pct': df.isna().mean().apply(lambda x: f"{x:.1%}"),
                          'missing_freq': df.isna().sum().apply(lambda x: f"{x:,.0f}").replace(',','.')})

def diagnosis(df_, var, vresp='survived', pred='pred', max_classes=5):
    
    df = df_.copy()
    
    if df[var].nunique()>max_classes:
        df[var] = pd.qcut(df[var], max_classes, duplicates='drop')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.pointplot(data=df, y=vresp, x=var, ax=ax1)
    sns.pointplot(data=df, y=pred, x=var, ax=ax1, color='red', linestyles='--', ci=None)
    
    ax2 = ax1.twinx()
    sns.countplot(data=df, x=var, palette='viridis', alpha=0.5, ax=ax2)
    ax2.set_ylabel('Frequência', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)
    
    plt.show()