## Importing Necessary Libraries
import pandas as pd
import numpy as np
import string
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



## Loading the data
columns = ['id','country','Label','Text']
data = pd.read_csv('Dataset/twitter_training.csv', names=columns)

### Data Preview
data.head()

data.shape

data.info()

data['Label'].value_counts()



## Data Preprocessing
data.dropna(inplace=True)

### Tokenization, StopWord Removal, Lemmatization
def preprocess(text):
    tokenized = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    stopwords_removed = [token for token in tokenized if token not in stop_words and token not in punctuation]

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in stopwords_removed]

    return " ".join(lemmatized)

data['Preprocessed Text'] = data['Text'].apply(preprocess)

### Label Columns in Numbers
#### (Irrelevant = 0, Negative = 1,  Nuetral = 2,  Positive = 3)
le_model = LabelEncoder()
data['Label'] = le_model.fit_transform(data['Label'])

### 'id' and 'country' columns are not required for analysis, So drop these Columns.
data = data.drop(['id', 'country'], axis = 1)

### Checking Null values
data.isnull().sum()



## Visualizing Data

### Distibution of Labels
#### (Irrelevant = 0, Negative = 1,  Nuetral = 2,  Positive = 3)
fig = plt.figure(figsize=(5,5))
sns.countplot(x='Label', data = data)
plt.xticks([0, 1, 2, 3], ['Irrelevant', 'Negative', 'Neutral', 'Positive'])
plt.show()



## Training the model
### Defining dependent and independent variable as x and y.
x = data['Preprocessed Text'].values
y = data['Label'].values

### Splitting the dataset into training set and testing set.
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42, stratify=data['Label'])

### Naive Bayes(Random Forest Classifier)
clf = Pipeline([
    ('vectorizer_tri_grams', TfidfVectorizer()),
    ('naive_bayes', (RandomForestClassifier()))
])

### Training
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

### Accuracy and Classification Report
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classifiation Report: \n\n", classification_report(y_test, y_pred))



## User Prediction
def predict_tweet_sentiment(tweet_text):
    cleaned_text = preprocess(tweet_text)

    prediction_idx = clf.predict([cleaned_text])

    prediction_label = le_model.inverse_transform(prediction_idx)

    return prediction_label[0]

while True:
    user_input = input("\nEnter a tweet to analyze (or type 'exit' to quit): ")

    if user_input.lower() in ['exit', 'quit']:
        print("Exiting...")
        break

    if user_input.strip() == "":
        continue

    sentiment = predict_tweet_sentiment(user_input)
    print(f"Predicted Sentiment: {sentiment}")



## Saving the Trained Model
import joblib

### Save the pipeline (includes vectorizer and Random Forest)
joblib.dump(clf, '/content/drive/MyDrive/Semester 5/Twitter Sentiment Analysis/sentiment_model.pkl')

### Save the Label Encoder (to convert numbers back to text labels)
joblib.dump(le_model, '/content/drive/MyDrive/Semester 5/Twitter Sentiment Analysis/label_encoder.pkl')

print("Model and Label Encoder saved successfully!")