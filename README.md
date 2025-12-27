🐦 Twitter Sentiment Analysis


📌 Overview

This project is a Machine Learning application designed to analyze and classify the sentiment of tweets. It uses Natural Language Processing (NLP) techniques to categorize text into Positive, Negative, Neutral, or Irrelevant sentiments. The project consists of a training module to build the model and a Streamlit web application for real-time user interaction.


🚀 Features

1. Data Preprocessing: text cleaning pipeline including tokenization, stopword removal, and lemmatization.

2. Model Training: Uses TF-IDF Vectorization and a Random Forest Classifier to achieve accurate predictions.

3. Interactive Web UI: A user-friendly interface built with Streamlit to input tweets and view results instantly.

4. Visual Feedback: Dynamic color-coded results (Green for Positive, Red for Negative, Blue for Neutral).

5. Transparency: View how the model "sees" your text (processed output) before prediction.

🛠️ Technology Stack

1. Language: Python

2. Web Framework: Streamlit

3. Machine Learning: Scikit-learn (RandomForest, TfidfVectorizer)

4. NLP: NLTK (Stopwords, WordNet, Punkt)

5. Data Handling: Pandas, NumPy

6. Visualization: Matplotlib, Seaborn

7. Model Persistence: Joblib


📂 Project Structure

├── Dataset

│   └── twitter_training.csv

├── Models

│   ├── sentiment_model.pkl

│   └── label_encoder.pkl

├── training.py

├── app.py

└── README.md


🏃‍♂️ Usage

1. Training the Model

If you want to retrain the model or update it with new data, run the training script. This will generate the .pkl files in your specified directory.
python training.py

Note: This script handles data loading, preprocessing, visualization of label distribution, and saving the model artifacts.

2. Running the Web App

To start the interface and make predictions: streamlit run app.py


The app will load the saved models (sentiment_model.pkl and label_encoder.pkl).

Enter a tweet in the text area and click "Analyze Sentiment".


🧠 Model Logic

The text data goes through the following pipeline:

1. Tokenization: Splitting text into words.

2. Filtering: Removing punctuation and English stopwords.

3. Lemmatization: Converting words to their root form (e.g., "running" -> "run").

4. Vectorization: Converting text to numbers using TF-IDF.

5. Classification: Predicting the category using Random Forest.


📊 Dataset

The model is trained on the twitter_training.csv dataset. Ensure this file is placed in the Dataset/ folder before running training.py.
