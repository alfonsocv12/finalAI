# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, nltk
from nltk import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Read the spam csv
tweets_df = pd.read_csv('AMLO.csv',index_col=None, na_values=['NA'])
tweets_df = tweets_df.drop(['id'],axis=1)

tweets_df_moreinfo = pd.read_csv('TAGS.csv', index_col=None, na_values=['NA'])

#Part dedicated to show df
# print(tweets_df.head(20))
# print(tweets_df_moreinfo.head(20))

'''
Nltk tokenizacion, remover malas palabras y stop_words
'''
# token = ToktokTokenizer()
stopwords = set(stopwords.words('spanish'))
with open('badwords.txt', 'r') as file:
    malas_palabras = (file.read().replace('\n', '')).split(',')
badwords = set(malas_palabras)
print(badwords)
