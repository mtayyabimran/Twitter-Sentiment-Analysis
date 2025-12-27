import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# NLTK Setup
# Streamlit may need to download these on the fly if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load PreTrained Models
@st.cache_resource # This caches the model so it loads only once
def load_models():
    model = joblib.load('Models/sentiment_model.pkl')
    encoder = joblib.load('Models/label_encoder.pkl')
    return model, encoder

try:
    clf, le_model = load_models()
except FileNotFoundError:
    st.error("Model files not found! Please make sure .pkl files are in the same directory.")
    st.stop()

# Preprocessing Function
def preprocess(text):
    if not text:
        return ""
    
    tokenized = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    stopwords_removed = [token for token in tokenized if token.lower() not in stop_words and token not in punctuation]

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in stopwords_removed]
    
    return " ".join(lemmatized)

# Streamlit UI
st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="🐦")

st.title("🐦 Twitter Sentiment Analysis")
st.write("Enter a tweet below to see if it's Positive, Negative, or Neutral.")

# User Input
user_input = st.text_area("Enter Tweet:", height=100, placeholder="e.g., I love this new feature!")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            # Preprocess the input
            cleaned_text = preprocess(user_input)
            
            # Predict
            prediction_idx = clf.predict([cleaned_text])
            prediction_label = le_model.inverse_transform(prediction_idx)[0]
            
            # Display Result
            st.markdown("---")
            st.subheader("Prediction Result:")
            
            # Dynamic coloring based on sentiment
            if "positive" in prediction_label.lower():
                st.success(f"**{prediction_label.upper()}** 😊")
            elif "negative" in prediction_label.lower():
                st.error(f"**{prediction_label.upper()}** 😠")
            else:
                st.info(f"**{prediction_label.upper()}** 😐")
                
            # Optional: Show processed text to user (debugging/transparency)
            with st.expander("See how the model sees your text"):
                st.write(f"**Processed Text:** {cleaned_text}")