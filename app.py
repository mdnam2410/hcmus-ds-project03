import streamlit as st
import numpy as np
from vncorenlp import VnCoreNLP
import joblib
import re
#vncorenlp_file = r'./VnCoreNLP/VnCoreNLP-1.1.1.jar'
#vncorenlp = VnCoreNLP(vncorenlp_file, annotators="wseg",  max_heap_size='-Xmx500m')

stopwords = []
with open("./vietnamese-stopwords.txt", 'r') as f:
  stopwords = list(map(lambda x: x.replace('\n', ''), f.readlines()))

def remove_stopwords(in_string):
  return " ".join([x for x in in_string.split(" ") if x not in stopwords])

def remove_punctuation(in_string):
    in_string = in_string.replace('\n', ' ')
    out_string = re.sub(r'[^\w\s]', '', in_string)

    return out_string
def tokenizer(in_string):
    in_string = remove_stopwords(remove_punctuation(in_string.lower()))
    #return vncorenlp.tokenize(in_string)[0]

trained_bayes = joblib.load("./TF-IDF_NaiveBayes.pkl")
def naive_bayes(news):
    # because this model return 0 for fake news and 1 for vice versa
    return 1 - trained_bayes.predict([news])

svm_model = joblib.load('SVM/SVM.pkl')

def logistic(news):
    return np.random.randint(0, 2)

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
st.selectbox('Choose model', ('Naive Bayes', 'Logistic Regression', 'SVM'), key='selectbox_model')
st.text_input('Enter news to validate', key='textinput_news')
if st.button('Predict', key='button_predict',):
    st.write(predict(st.session_state.textinput_news, st.session_state.selectbox_model))
