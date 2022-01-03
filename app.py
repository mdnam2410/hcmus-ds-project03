import streamlit as st
import numpy as np

def naive_bayes(news):
    return np.random.randint(0, 2)

def logistic(news):
    return np.random.randint(0, 2)

def predict(news, model):
    m = naive_bayes if model == 'Naive Bayes' else logistic
    return f"Result: {m(news)}"

st.title('Fake news detector')
st.selectbox('Choose model', ('Naive Bayes', 'Logistic Regression'), key='selectbox_model')
st.text_input('Enter news to validate', key='textinput_news')
if st.button('Predict', key='button_predict',):
    st.write(predict(st.session_state.textinput_news, st.session_state.selectbox_model))
