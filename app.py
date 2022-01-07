import streamlit as st
import numpy as np
import pandas as pd
from vncorenlp import VnCoreNLP
import joblib
import re
vncorenlp_file = r'./VnCoreNLP/VnCoreNLP-1.1.1.jar'

# Call this function once
@st.cache(hash_funcs={VnCoreNLP: id})
def get_vncorenlp():
    return VnCoreNLP(vncorenlp_file, annotators='wseg', max_heap_size='-Xmx500m')


# ---- Naive Bayes model ----

stopwords = []
with open("./vietnamese-stopwords.txt", 'r') as f:
  stopwords = list(map(lambda x: x.replace('\n', ''), f.readlines()))

def remove_stopwords(in_string):
  return " ".join([x for x in in_string.split(" ") if x not in stopwords])

def remove_punctuation(in_string):
    in_string = in_string.replace('\n', ' ')
    out_string = re.sub(r'[^\w\s]', '', in_string)
    return out_string

vncorenlp = get_vncorenlp()
def tokenizer(in_string):
    in_string = remove_stopwords(remove_punctuation(in_string.lower()))
    return vncorenlp.tokenize(in_string)[0]

trained_bayes = joblib.load("./TF-IDF_NaiveBayes.pkl")
def naive_bayes(news):
    return trained_bayes.predict([news])


# ---- Logistic regression model ----

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

vncorenlp_tokenizer = vncorenlp
logistic_stopwords = logistic_get_stopwords()
logistic_model = logistic_get_model()
def logistic(news):
    return logistic_model.predict([news])


# ---- SVM model ----

svm_model = joblib.load('SVM/SVM.pkl')
def svm(news):
    return svm_model.predict([news])

def predict(news, model):
    if model == 'Naive Bayes':
        m = naive_bayes
    elif model == 'Logistic Regression':
        m = logistic
    else:
        m = svm
    return f"Result: {'This is a fake news' if m(news)[0] == 1 else 'This is a fact news' }"

st.title('Fake news detector')
st.header('Authors')
st.table(pd.DataFrame(
    data={
        'ID': ['19120080', '19120298', '19120460'],
        'Name': ['Lê Đức Huy', 'Mai Duy Nam', 'Nguyễn Hữu Bình']
    },
))
st.header('Models')
st.selectbox('Choose model', ('Naive Bayes', 'Logistic Regression', 'SVM'), key='selectbox_model')
st.text_input('Enter news to validate', key='textinput_news')
if st.button('Predict', key='button_predict',):
    st.write(predict(st.session_state.textinput_news, st.session_state.selectbox_model))
