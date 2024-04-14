import pandas as pd
import streamlit as st
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from PIL import Image  # Import the Image module
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import math
from io import BytesIO 
import base64
import plotly.express as px

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess tweets
def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(tweet)
    stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words("english"))
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords_set]
    return ' '.join(tokens)

# Load the dataset
csv_file_path = r'C:\Users\Sai Chaturya\Downloads\Sentiment-Analysis-with-Naive-Bayes-Streamlit-main\Sentiment-Analysis-with-Naive-Bayes-Streamlit-main\data\train.csv'
df = pd.read_csv(csv_file_path)
df = df.drop(columns=['selected_text'])
df["text"] = df["text"].astype(str)

# Create a TF-IDF vectorizer and Naive Bayes classifier pipeline
model = make_pipeline(TfidfVectorizer(preprocessor=preprocess_tweet), MultinomialNB())

# Split the data into features (X) and target (y)
X = df['text']
y = df['sentiment']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Define the classify_tweet_with_scores function
def classify_tweet_with_scores(tweet):
    predicted_sentiment = model.predict([tweet])[0]
    sentiment_scores = model.predict_proba([tweet])[0]
    return predicted_sentiment, sentiment_scores

# Define the main function
def main():
    # Add a banner image
    banner_image = Image.open("C:/Users/Sai Chaturya/Downloads/Sentiment-Analysis-with-Naive-Bayes-Streamlit-main/Sentiment-Analysis-with-Naive-Bayes-Streamlit-main/images/sentimentanalysishotelgeneric-2048x803-1.jpg")

    st.image(banner_image, use_column_width=True)

    st.title("Sentiment Analysis with Naïve Bayes")

    # User input for text
    user_input = st.text_area("Enter a text:", "I love this product!", key="text_input")

    if st.button("Classify Sentiment", key="classify_button"):
        predicted_sentiment, sentiment_scores = classify_tweet_with_scores(user_input)
        image_size = (100, 100) 

        # Display an image based on the sentiment
        if predicted_sentiment == "positive":
            image = Image.open("images/positive.jpg")
        elif predicted_sentiment == "negative":
            image = Image.open("images/negative.jpg")
        else:
            image = Image.open("images/neutral.jpg")

        resized_image = image.resize(image_size, Image.BILINEAR)

        # Convert the image to base64
        buffered = BytesIO()
        resized_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Display resized image with markdown spacer for center alignment
        st.markdown(
            f'<p align="center"><img src="data:image/png;base64,{img_str}" alt="{predicted_sentiment}" width="{image_size[0]}"></p>',
            unsafe_allow_html=True
        )

        # Display predicted sentiment in uppercase
        st.markdown(f'<p align="center"><b>{predicted_sentiment.upper()}</b></p>', unsafe_allow_html=True)

        # Display sentiment scores in a table
        scores_df = pd.DataFrame(sentiment_scores, columns=['Negative', 'Neutral', 'Positive'])
        st.write("Sentiment Scores:")
        st.table(scores_df)

if __name__ == "__main__":
    main()
