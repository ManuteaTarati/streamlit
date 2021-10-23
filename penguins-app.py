import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
import catboost 
import xgboost as xgb

#cd "Documents\Projets Data Science perso\streamlit"
#streamlit run test.py

st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species! 

Author: Manutea TARATI (manuts@hotmail.fr)
""")

st.sidebar.header('User Input Features')

def user_input_features():
    island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
    sex = st.sidebar.selectbox('Sex',('male','female'))
    bill_length_mm = st.sidebar.slider('Bill lenght (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    data = {'island':island,
            'sex':sex,
            'bill_length_mm':bill_length_mm,
            'bill_depth_mm':bill_depth_mm,
            'flipper_length_mm':flipper_length_mm,
            'body_mass_g':body_mass_g
            }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.sidebar.subheader("Model")
classifier = st.sidebar.selectbox('Classifier', ('RandomForestClassifier','XGBClassifier','LightGBMClassifier', 'CatBoostClassifier'))



penguins_raw = pd.read_csv("data/penguins_cleaned.csv")
penguins = penguins_raw.drop(columns=["species"])
penguins = pd.concat([input_df, penguins])
penguins = pd.get_dummies(penguins)



X = penguins[1:].reset_index(drop=True)
X_user_input = penguins[:1]
target_mapper = {"Adelie":0, "Chinstrap":1, "Gentoo":2}
y = penguins_raw["species"]
y = y.map(target_mapper)

st.subheader('User Input Features')
st.write(input_df)
st.write(f"Model selected: {classifier}")

if classifier == "RandomForestClassifier":
    clf = RandomForestClassifier()
elif classifier == "XGBClassifier":
    clf = xgb.XGBClassifier()
elif classifier == 'LightGBMClassifier':
    clf = lgb.LGBMClassifier()
elif classifier == 'CatBoostClassifier':
    clf = catboost.CatBoostClassifier()
    

clf.fit(X,y)

prediction = clf.predict(X_user_input)
prediction_proba = clf.predict_proba(X_user_input)

st.subheader('Prediction')
penguins_species = np.array(["Adelie","Chinstrap", "Gentoo"])
st.write(penguins_species[prediction])

st.subheader('Prediction probabilities')
st.write(prediction_proba)

st.subheader('Model evaluation')
st.write(f"Accuracy: {clf.score(X,y)*100}%")
st.write(f"Confusion matrix")
st.write(confusion_matrix(y, clf.predict(X)))
