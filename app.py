import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Make sure resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    # Lowercase
    text = text.lower()

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Keep only alphanumeric tokens
    tokens = [t for t in tokens if t.isalnum()]

    # Remove stopwords and punctuation
    tokens = [t for t in tokens if t not in stopwords.words('english') and t not in string.punctuation]

    # Stemming
    tokens = [ps.stem(t) for t in tokens]

    return " ".join(tokens)


tfidf=pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model_spam.pkl', 'rb'))

st.title("spam classifier ")

input_sms=st.text_area("enter your message")

if st.button("predict"):

    # 1. preprocess
    transform_sms=tfidf.transform([input_sms])
    # 2. vectorize
    vectorize_inout=tfidf.transform([input_sms])
    # 3. predict
    result= model.predict(vectorize_inout)[0]
    # 4. Display
    if result==1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")
