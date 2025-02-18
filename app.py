import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import nltk
import streamlit as st

nltk.download('vader_lexicon')

# Function to perform sentiment analysis and generate plots
def analyze_sentiment(api_key, user_input, buy_threshold):
    # Initialize sentiment_results list
    sentiments = []

    # Function to format the query for NewsAPI
    def format_query(ticker):
        if '.' in ticker:
            base_ticker = ticker.split('.')[0]
        else:
            base_ticker = ticker
        return base_ticker

    # Fetch news headlines and perform sentiment analysis for the given company ticker
    if user_input:
        # Format the query
        formatted_query = format_query(user_input)
        
        # Define the API endpoint URL
        url = f"https://newsapi.org/v2/everything?q={formatted_query}&language=en&apiKey={api_key}"

        # Make a GET request to the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response JSON
            data = response.json()

            # Get the news articles
            articles = data.get('articles', [])

            if articles:
                # Perform sentiment analysis on each article headline
                sid = SentimentIntensityAnalyzer()
                for article in articles:
                    headline = article.get('title', '')
                    scores = sid.polarity_scores(headline)
                    sentiment = scores['compound']
                    sentiments.append(sentiment)

                # Generate a line plot for the sentiments
                fig, ax = plt.subplots()
                ax.plot(sentiments, color='blue')
                ax.axhline(0, color='black', linestyle='--')
                ax.set_title(f'Sentiment Analysis of Stock {user_input} News Headlines')
                ax.set_xlabel('Headline')
                ax.set_ylabel('Sentiment Polarity')

                # Add markers to indicate buying decision
                buy_markers = [i for i, sentiment in enumerate(sentiments) if sentiment >= buy_threshold]
                ax.plot(buy_markers, np.array(sentiments)[buy_markers], 'go', markersize=8, label='Buy')
                ax.legend()

                # Display the plot
                st.pyplot(fig)

                # Calculate histogram values
                num_buy = len(buy_markers)
                num_not_buy = len(sentiments) - num_buy

                # Display histogram
                fig_hist, ax_hist = plt.subplots()
                ax_hist.bar(['Buy', 'Not Buy'], [num_buy, num_not_buy], color=['green', 'red'])
                ax_hist.set_title('Distribution of Buy vs Not Buy Sentiments')
                ax_hist.set_xlabel('Sentiment Category')
                ax_hist.set_ylabel('Count')
                st.pyplot(fig_hist)

            else:
                st.warning(f'No news articles found for ticker: {user_input}')
        else:
            st.error('Failed to fetch news articles. Please check your API key and internet connection.')

# Streamlit app layout
st.title('Stock News Sentiment Analysis')

# Sidebar for input fields
with st.sidebar:
    st.subheader('Input Parameters')
    api_key = st.text_input('Enter your NewsAPI API key', '')
    user_input = st.text_input('Enter Stock Ticker', 'GOOG')
    buy_threshold = st.slider('Buy Threshold', min_value=0.0, max_value=1.0, value=0.1, step=0.01)

# Main content area for displaying graphs
st.subheader('Sentiment Analysis Results')

# Call the function to perform sentiment analysis and generate plots
if api_key and user_input:
    analyze_sentiment(api_key, user_input, buy_threshold)
