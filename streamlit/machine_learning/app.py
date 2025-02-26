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

data = pd.read_csv('mushrooms.csv')

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

    df = load_data()

    if st.sidebar.checkbox('Show raw data', False):
        st.subheader('Mushroom Data Set (Classification)')
        st.write(df)

        






if __name__ == '__main__':
    main()