# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, nltk
from nltk import sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB, BernoulliNB

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
stemmer = SnowballStemmer('spanish')

def preprocesing(log):
    '''
    Funcion encargada de remover
    las malas palabras
    '''
    log_remove = []
    for word in log:
        if word in badwords:
            log_remove.append('MALAPALABRA')
        elif word not in stop_words:
            log_remove.append(stemmer.stem(word))
            # log_remove.append(word)
    return log_remove

tweets_df['log'] = tweets_df.apply(lambda tweets_df: nltk.word_tokenize(tweets_df['log'].lower()), axis=1)
tweets_df['log'] = tweets_df['log'].apply(lambda log: preprocesing(log))
tweets_df['log'] = tweets_df['log'].apply(lambda log: [word for word in log if not word.isdigit()])

'''
vectorize df
'''
vectorize = CountVectorizer(lowercase=False, tokenizer=(lambda arg: arg), preprocessor=None)
data_features = vectorize.fit_transform(tweets_df['log'])
# print(vectorize.vocabulary_)

'''
Train test split
'''
data_labels = tweets_df['positivo_negativo'].values

# X_train, X_test, Y_train, Y_test = train_test_split(data_features.toarray(), data_labels, test_size=0.3)
X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_labels, test_size=0.3)
'''
Using MultinomialNB model
'''
clf = MultinomialNB()
# clf = GaussianNB()
# clf = ComplementNB()
# clf = BernoulliNB()

clf.fit(X_train, Y_train)

# score = clf.score(X_test,Y_test)
# print(score,'\n')

'''
Aqui es donde se procesaron los datos que tienen
mas informacion
'''
tweets_df_moreinfo = tweets_df_moreinfo.drop(['geo_coordinates','id_str','profile_image_url', 'from_user',\
                                              'in_reply_to_user_id_str','in_reply_to_screen_name','in_reply_to_status_id_str',\
                                              'source','user_friends_count','entities_str'], axis=1)

tweets_df_moreinfo['user_lang'] = tweets_df_moreinfo['user_lang'].apply(lambda row: row if row == 'es' else None)

tweets_df_moreinfo = tweets_df_moreinfo.dropna(subset=['log','user_lang'])

# print(tweets_df_moreinfo.head(20),'\n')
# print(tweets_df_moreinfo.describe(),'\n')
# # print(tweets_df_moreinfo.apply(lambda x: x.isnull().any()),'\n')
# print(pd.DataFrame({'percent_missing': tweets_df_moreinfo.isnull().sum() * 100 / len(tweets_df_moreinfo)}),'\n')
# print(pd.DataFrame({'percent_unique': tweets_df_moreinfo.apply(lambda log: log.unique().size/log.size*100)}),'\n')

tweets_df_moreinfo['log_proces'] = tweets_df_moreinfo.apply(lambda tweets_df_moreinfo: nltk.word_tokenize(tweets_df_moreinfo['log'].lower()), axis=1)
tweets_df_moreinfo['log_proces'] = tweets_df_moreinfo['log_proces'].apply(lambda log: preprocesing(log))
tweets_df_moreinfo['log_proces'] = tweets_df_moreinfo['log_proces'].apply(lambda log: [word for word in log if not word.isdigit()])

'''
Se vectoriza para poder
procesarlo
'''
data_features_2 = vectorize.transform(tweets_df_moreinfo['log_proces'])
# print(vectorize_2.vocabulary_)
'''
Se predice y se guarda la informacion
en un csv para despues obtener mas
informacion de ese df
'''

# print(X_train,'\n')
# print(data_features_2,'\n')
predictions = clf.predict(data_features_2)
tweets_df_moreinfo = tweets_df_moreinfo.drop(['user_lang','log_proces'], axis=1)
tweets_df_moreinfo['positivo_negativo'] = predictions
# print(data_frame2.head(20),'\n')
# tweets_df_moreinfo.to_csv(path_or_buf='newdf.csv')
