import os
import re
import tweepy as tw
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
import tensorflow as tf



#Basic input to fill for the code to works
consumer_key = ""
consumer_secret_key = ""
access_token_key = ""
access_secret_token_key = ""
key_words = "Bitcoin"
date_until = "2021-10-15"


#Authantification
auth = tw.OAuthHandler(consumer_key, consumer_secret_key)
auth.set_access_token(access_token_key, access_secret_token_key)
api = tw.API(auth, wait_on_rate_limit=True)


#Search with the Twitter API the tweets given a keyword and a date until you want the tweets
def search(key_words, date_until):
    query = tw.Cursor(api.search_tweets, q=key_words, lang="en", until=date_until).items(10)
    tweet = [[tweet.created_at, tweet.text] for tweet in query]
    tweet_df = pd.DataFrame(data=tweet, 
                    columns=['created_at', 'text'])
    return tweet_df


 #Clean DataFrame   
def clean_df(tweet_df):
    
    tweet_txt_df = []

    for tweet in tweet_df['text'].values:
        if type(tweet) == np.float64:
            return ""
        twt_txt = tweet.lower()
        twt_txt = re.sub("'", "", twt_txt)
        twt_txt = re.sub("@[A-Za-z0-9_]+","", twt_txt)
        twt_txt = re.sub("#[A-Za-z0-9_]+","", twt_txt)
        twt_txt = re.sub(r'http\S+', '', twt_txt)
        twt_txt = re.sub('[()!?]', ' ', twt_txt)
        twt_txt = re.sub('\[.*?\]',' ', twt_txt)
        twt_txt = re.sub("[^a-z0-9]"," ", twt_txt)
        twt_txt = twt_txt.split()
        stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from", "rt"]
        twt_txt = [w for w in twt_txt if not w in stopwords]
        twt_txt = " ".join(word for word in twt_txt)
        tweet_txt_df.append(twt_txt)
        tweet_df['Clean_text'] = pd.Series(tweet_txt_df)
        tweet_df['Clean_text'].drop_duplicates(inplace=True)
        clean_tweet_df = tweet_df[['created_at', 'Clean_text']]        

    return clean_tweet_df


#NLP Model using BERT basic model








#Main function to call
def main():
    df_1 = search(key_words, date_until)
    df_2 = clean_df(df_1)
    print(df_2)


if __name__ == "__main__":
    main()