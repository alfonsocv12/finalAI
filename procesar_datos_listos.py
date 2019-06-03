import pandas as pd, numpy as np, matplotlib, matplotlib.pyplot as plt, nltk,json
from nltk.stem import SnowballStemmer

#se le el data frame
# tweets_df = pd.read_csv('tags_predecido.csv',index_col=None, na_values=['NA'])
tweets_df = pd.read_csv('tags_predecido_varias_fechas.csv',index_col=None, na_values=['NA'])
tweets_df = tweets_df.drop(['id','from_user_id_str','status_url'],axis=1)

'''
Procesar user location
para que se repitan un poco mas
los valores
'''
vector_locations = {}
vector_locations['cdmx'] = ['méxico','mexico','distrito','ciudad','d.f.',\
                            'mèxico','df','cdmx\n','méxico..','mex','mëxico',\
                            '¡méxico!','cd.mex','mx','mexicod.f.','méxico/...',\
                            'cd.','estado','cd']
vector_locations['nan'] = [' ','','️','aqui','.....',\
                           'ningunlado','vivo','𝕬𝖖𝖚í\xa0𝖒𝖊𝖗𝖔\xa0','aquí','anywhere',\
                           'somewhere','https://lopezobrador.org.mx/2018/08/17/documentos-de-consulta-sobre-aeropuerto/','earth','ccs','casa',\
                           't.a.r.d.i.s', 'werolandia','¡en','stuck','una',\
                           'trc', 'n23.16', 'the', 'd.', 'de',\
                           'en','iv', 'üt:','el','la'\
                           'g77x']

vector_locations['chihuahua'] = ['chihuahuamexico','chihuahua']

def mexico_filtro(location):
    '''
    Funcion encargada
    de filtrar variaciones de cdmx
    '''
    if location in vector_locations['cdmx']:
        return 'cdmx'
    elif location in vector_locations['nan']:
        return 'nan'
    elif location in vector_locations['chihuahua']:
        return 'chihuahua'
    else:
        return location

tweets_df.user_location = tweets_df['user_location'].apply(lambda location: (str(location).lower()).replace(',','').split(' ')[0])
tweets_df.user_location = tweets_df['user_location'].apply(lambda location: mexico_filtro(location))

def group_plot(features, target):
    '''
    Funcion encargada de hacer una group_plot
    '''
    ind = np.arange(len(features.unique()))
    width = 0.35
    fig, ax = plt.subplots()

def bar_plot_dic(dic):
    '''
    Funcion encargada de graficar
    dicionarios en una grafica de
    barras
    '''
    plt.bar(range(len(dic)), list(dic.values()), align='center')
    plt.xticks(range(len(dic)), list(dic.keys()))
    plt.show()

# group_plot(tweets_df.user_location, tweets_df.positivo_negativo)
positivo_negativo = {'positivo':sum(tweets_df['positivo_negativo'] > 0),
                     'negativo':sum(tweets_df['positivo_negativo'] == 0)}
# bar_plot_dic(positivo_negativo)
# print('positivo: {} \nnegativo: {}'.format(sum(tweets_df['positivo_negativo'] > 0), sum(tweets_df['positivo_negativo'] == 0)))

vector_locations_suma = {}
for location in tweets_df.user_location.unique():
    vector_locations_suma[location] =sum(tweets_df['user_location'] == location)

# bar_plot_dic(vector_locations_suma)
# print(sorted(vector_locations_suma.items(),key=lambda kv: kv[1], reverse=True))
# print('positivo: {}'.format(positivo_negativo['positivo']),'\n','negativo: {}')
# print(tweets_df.head(20),'\n')
# print(tweets_df.describe(),'\n')
# print(pd.DataFrame({'percent_unique': tweets_df.apply(lambda log: log.unique().size/log.size*100)}),'\n')
# print(tweets_df.user_location.unique(),'\n')

'''
positivo_negativo en base al
tiempo
'''
tweets_df['created_at_split'] = tweets_df.created_at.apply(lambda date: date.split(' '))
tweets_df['dia'] = tweets_df.created_at_split.apply(lambda date: date[2])
tweets_df = tweets_df.drop(['created_at_split'], axis=1)

vector_dia = {}
for dia_num in tweets_df.dia.unique():
    vector_dia[dia_num] = sum(tweets_df.dia == dia_num)

# print(vector_dia)
# bar_plot_dic(vector_dia)
# sorted_dia = sorted(vector_dia.items(),key=lambda kv: kv[1], reverse=True)
# for dia in sorted_dia:
#     print('dia {} se tienen {} tweets'.format(dia[0],dia[1]))

vector_positivo_negativo_dia = {}
for dia_num in tweets_df.dia.unique():
    vector_positivo_negativo_dia['dia {}'.format(dia_num)] = {'cantidad':sum(tweets_df['dia'] == dia_num),
                                                              'positivo':sum((tweets_df['dia'] == dia_num) & (tweets_df['positivo_negativo'] > 0)),
                                                              'negativo':sum((tweets_df['dia'] == dia_num) & (tweets_df['positivo_negativo'] == 0))}
# print(json.dumps(vector_positivo_negativo_dia, indent=4, sort_keys=True))
# print(tweets_df['dia'].head(20))
# print(pd.DataFrame({'percent_unique': tweets_df.apply(lambda log: log.unique().size/log.size*100)}),'\n')
# print(tweets_df.dia.unique(),'\n')

dia_dataframe = pd.DataFrame(vector_positivo_negativo_dia)
# print(dia_dataframe.index)
# print(dia_dataframe.columns)
pos = list(range(len(dia_dataframe.columns)))
width = 0.25

fig, ax = plt.subplots(figsize=(10,5))
# count = 0
# array_colores = ['#EE3224','#F78F1E','#FFC222']
# columns = ['dia 27', 'dia 28', 'dia 30', 'dia 31']
# for index_c in dia_dataframe.index:
#     array_valores = []
#     for valor in dia_dataframe.loc[index_c,:]:
#         array_valores.append(valor)
#     print(pos)
#     count += 1
# print(dia_dataframe.columns[0])

plt.bar(pos, dia_dataframe.loc['cantidad',:], width, alpha=0.5, color='#EE3224',label='cantidad')
plt.bar([p + width for p in pos], dia_dataframe.loc['positivo',:], width, alpha=0.5, color='#F78F1E',label='positivo')
plt.bar([p + width*2 for p in pos], dia_dataframe.loc['negativo',:], width, alpha=0.5, color='#FFC222',label='negativo')

ax.set_ylabel('Cantidad')
ax.set_title('Dias')
ax.set_xticks([p + 1.5 * width for p in pos])
ax.set_xticklabels(dia_dataframe.columns)
plt.xlim(min(pos)-width, max(pos)+width*5)
plt.ylim([0, max(dia_dataframe.loc['cantidad',:] +\
                 dia_dataframe.loc['positivo',:] +\
                 dia_dataframe.loc['negativo',:])])
plt.legend(dia_dataframe.index, loc='upper left')
plt.grid()
plt.show()
