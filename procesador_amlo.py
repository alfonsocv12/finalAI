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
Aqui se carga nuestro txt con las
malas palabras y se pone en un
objeto set como las stopwords
'''
stop_words = set(stopwords.words('spanish'))

with open('badwords.txt', 'r') as file:
    malas_palabras = (file.read().replace('\n', '')).lower().split(',')

badwords = set(malas_palabras)

'''
Aqui se comienza a aplicar la tokenizacion
remover stopwords y procesamiento de
malas palabras
'''

def remove_badword_stopwords(log):
    '''
    Funcion encargada de remover
    las malas palabras
    '''
    log_remove = []
    for word in log:
        if word in badwords:
            log_remove.append('MALAPALABRA')
        elif word not in stop_words:
            log_remove.append(word)
    return log_remove

tweets_df['log'] = tweets_df.apply(lambda tweets_df: nltk.word_tokenize(tweets_df['log'].lower()), axis=1)
tweets_df['log'] = tweets_df['log'].apply(lambda log: remove_badword_stopwords(log))

'''
vectorize df
'''
vectorize = CountVectorizer(lowercase=False, tokenizer=(lambda arg: arg), preprocessor=None)
data_features = vectorize.fit_transform(tweets_df['log'])

# print all the words that are include on the df
# print(vectorize.vocabulary_)

# summarize encoded vector
# print(data_features.shape)
# print()
# print(data_features.toarray())
# print()

'''
Train test split
'''
data_labels = tweets_df['positivo_negativo'].values

X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_labels, test_size=0.3)

'''
Using MultinomialNB model
'''

clf = MultinomialNB()
clf.fit(X_train, Y_train)

score = clf.score(X_test, Y_test)
print(score)
