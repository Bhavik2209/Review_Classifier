import pickle
import pandas as pd
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and tokenizer within the main function
@st.cache(allow_output_mutation=True)
def load_model():
    pickle_in = open(r"C:\Users\SVI\Desktop\text_reog\clf.pkl", "rb")
    reg = pickle.load(pickle_in)

    pickle_tra = open(r'C:\Users\SVI\Desktop\text_reog\tra.pkl', 'rb')
    tfidf = pickle.load(pickle_tra)
    return reg, tfidf


def tokenizer_porter(text):
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]

def predict_review(model, vectorizer, user_data):
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    user_df = vectorizer.transform([user_data])
    prediction = model.predict(user_df)
    return "good review" if prediction == 1 else "bad review"

def main():
    st.title("Performance Index")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Performance App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Load the model and vectorizer
    reg, tfidf = load_model()

    user_data = st.text_input("Enter your Review")
    result = ""

    if st.button("Predict"):
        result = predict_review(reg, tfidf, user_data)

    st.success(f'Your review is classified as a {result}')

if __name__ == '__main__':
    main()
