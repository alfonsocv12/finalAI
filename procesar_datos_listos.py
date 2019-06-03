import pandas as pd

#se le el data frame
tweets_df = pd.read_csv('tags_predecido.csv',index_col=None, na_values=['NA'])
tweets_df = tweets_df.drop(['id'],axis=1)

print(tweets_df.head(20),'\n')
