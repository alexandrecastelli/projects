import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title('Binary Classification Web App')
    st.sidebar.title('Binary Classification Web App')
    st.markdown('Are your mushrooms edible or poisonous? 🍄')
    st.sidebar.markdown('Are your mushrooms edible or poisonous? 🍄')

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache_data(persist=True)
    def split(df):
        y = df.type
        X = df.drop(columns=['type'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        return X_train, X_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            ConfusionMatrixDisplay(model, X_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            RocCurveDisplay(model, X_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            PrecisionRecallDisplay(model, X_test, y_test, display_labels=class_names)
            st.pyplot()

    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader('Choose Classifier')
    classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine (SVM)', 'Logistic Regression', 'Random Forest'))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader('Model Hyperparameters')
        C = st.sidebar.number_input('C (Regularization parameter)', 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio('Kernel', ('rbf', 'linear'), key='kernel')
        gamma = st.sidebar.radio('Gamma (Kernel Coefficient)', ('scale', 'auto'), key='gamma')









    if st.sidebar.checkbox('Show raw data', False):
        st.subheader('Mushroom Data Set (Classification)')
        st.write(df)

        






if __name__ == '__main__':
    main()