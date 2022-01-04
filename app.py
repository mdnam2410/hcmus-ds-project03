import streamlit as st
import numpy as np
from vncorenlp import VnCoreNLP
import joblib
import re

@st.cache(hash_funcs={VnCoreNLP: id})
def get_vncorenlp():
    return VnCoreNLP(r'./VnCoreNLP/VnCoreNLP-1.1.1.jar', annotators='wseg', max_heap_size='-Xmx500m')

def naive_bayes(news):
    return np.random.randint(0, 2)

def logistic_preprocess(text: str):
    # Lowercase
    text = text.lower()

    # Remove URL
    text = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '', text)

    # Remove HTML entity
    text = re.sub(r'(&\w+;|&#\d+;)', '', text)

    # Remove punctuations
    text = re.sub(r'[^A-Za-z0-9àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ\s.,_-]', '', text)
    
    return text

def logistic_tokenize(text: str):
    return [token for sublist in vncorenlp_tokenizer.tokenize(text) for token in sublist]

@st.cache
def logistic_get_stopwords():
    return joblib.load('logistic-regression-model/logistic_stopwords.pkl')

def logistic_get_model():
    return joblib.load('logistic-regression-model/logistic_model.pkl')


vncorenlp_tokenizer = get_vncorenlp()
logistic_stopwords = logistic_get_stopwords()
logistic_model = logistic_get_model()
def logistic(news):
    return logistic_model.predict([news])[0]

def predict(news, model):
    m = naive_bayes if model == 'Naive Bayes' else logistic
    return f"Result: {m(news)}"

st.title('Fake news detector')
st.selectbox('Choose model', ('Naive Bayes', 'Logistic Regression'), key='selectbox_model')
st.text_input('Enter news to validate', key='textinput_news')
if st.button('Predict', key='button_predict',):
    st.write(predict(st.session_state.textinput_news, st.session_state.selectbox_model))
