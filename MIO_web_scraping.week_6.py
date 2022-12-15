#!/usr/bin/env python
# coding: utf-8

# In[148]:


import spotipy
import random
import numpy as np
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials


# In[149]:


c_id = "y c40f03f084aa4e81b635c7f5137e089e"
c_se ="yd96a58a623924628b64eb01523285f8c"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=c_id, client_secret=c_se))


# In[189]:


import sys
get_ipython().system('conda install --yes --prefix {sys.prefix} -c anaconda beautifulsoup4')
get_ipython().system('conda install --yes --prefix {sys.prefix} -c anaconda requests')


# In[190]:


#Create a function to scrape the Billboards 100 HOT songs and create a local dataframe of songs with them including:
#Song’s name
#Song’s artist


# In[191]:


from bs4 import BeautifulSoup


# In[192]:



# 1. import libraries
from bs4 import BeautifulSoup
import requests
import pandas as pd


# In[193]:


# 2. find url and store it in a variable
url = "https://www.imdb.com/chart/top"


# In[194]:


# 3. download html with a get request 
response = requests.get(url)


# In[195]:


response.status_code # 200 status code means OK!


# In[196]:


####exercise---------------------------------------------------------------------------------------------

url="https://www.billboard.com/charts/hot-100/"


# In[197]:


url


# In[198]:


soup = BeautifulSoup(url, 'html.parser') 


# In[199]:


# 3. download html with a get request 
response = requests.get("https://www.billboard.com/charts/hot-100/")


# In[200]:


response #200 status code means OK!


# In[201]:


response.content


# In[202]:


pip install html5lib


# In[203]:


from bs4 import BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')
print(soup.title)


# In[204]:


soup.select("li > h3")[0].get_text().strip("\t \n")


# In[205]:


soup.select("li > span")[0].get_text().strip("\t \n")


# In[ ]:





# In[167]:


#### song name list
song_list_name=[]

for i in range(100):
    song_list_name.append(soup.select("li > h3")[i].get_text().strip("\t \n"))

song_list_name


# In[168]:


# song name it works
shit_len= range(len(soup.select("li > h3")[0]),100)
list_name=[]

for i in shit_len:
    list_name.append(soup.select("li  h3")[i].get_text().strip("\t \n"))

list_name


# In[169]:


soup.select("li > h3")


# In[170]:


#for i como usarlo
viejalista = [0,1,2,3,4]
nuevalista =[]

for i in viejalista:
    nuevalista.append(i**2)
nuevalista


# In[171]:


### artist name 
soup.select("li > h3")[0].find_next("span").get_text().strip() 
# the first span belongs to the index number, you must add find_next


# In[172]:


#artist name list
artist_list_name=[]

for i in range(100):
    artist_list_name.append(soup.select("li > h3")[i].find_next("span").get_text().strip())

artist_list_name


# In[173]:


df = pd.DataFrame()  #creating data frame
  


# In[174]:


df["song_name"]=song_list_name  #adding lists as column into the data frame
df["artist_name"]=artist_list_name


# In[175]:


df


# In[176]:


# user gives name of song and tells if its in top from the last table


# In[177]:


x = input('Enter your name:')  #input part
print('Hello, ' + x)


# In[178]:


#finding f perfect macth

(df['song_name'].eq('Anti-Hero')).any()


# In[179]:


#check if partial match   # it doesnt work with the espace

df['song_name'].str.contains('Anti Hero').any()


# In[180]:


## how many times are similar words

df['song_name'].str.contains('anti').sum()


# In[406]:


x = input('Introduce song name:')
y = random.choice(df['song_name'])


if (df['song_name'].eq(x)).any() == True:
    new_list= list(df['song_name'])
    print("This song is in top 100. \nHey! This other one is in TOP100 as well  ")
    
elif df['song_name'].str.contains(x).any():
    print("It seems you have a typo, try again. N° of songs with similar name: "+ str(df['song_name'].str.contains(x).sum()))
else:
    print("Something went wrong")
    


# In[401]:


def top100():  # fomrula javi quitar cancion de la lista
    x = random.choice(df['song_name'])
    user_choice = input('Please, enter a TOP100 song from Billboard: ')
    
    if user_choice not in list(df['song_name']):
        print('Sorry, this song is not included in the TOP100. Try with another song')
    else:
        print('This song is hot!', user_choice, 'Try with this song from the TOP100', x)
        new_df = list(df['song_name'])
        new_df.remove(user_choice)
top100()


# In[183]:


def top100():   ## formula sara random part
    x = random.choice(df['song_name'])
    user_choice = input('Please, enter a TOP100 song from Billboard: ')
    
    if user_choice not in list(df['song_name']):
        print('Sorry, this song is not included in the TOP100. Try with another song')
    else:
        print('If you like', user_choice, 'this is in TOP100 as well', x)
top100()


# In[184]:


import difflib  ###similar formula to contains and with the same effect (no space or capital letters)

word = "lea"
possibilities = ["love", "learn", "lean", "moving", "hearing"]
n = 3
cutoff = 0.7

close_matches = difflib.get_close_matches(word, 
                possibilities, n, cutoff)

print(close_matches)


# In[206]:


#check if exact string 'Eas' exists in conference column
(df['song_name'].eq('Eas')).any()

#check if partial string 'Eas' exists in conference column
df['song_name'].str.contains('Eas').any()


# In[186]:


############ miercoles


# In[207]:


import config


# In[208]:



import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials



#Initialize SpotiPy with user credentias
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id= config.client_id,
                                                           client_secret= config.client_secret))

# The "sp" variable has two useful funtions:
# The first usefull function is:
# .search(q='',limit=n)
# .search(q="track:"+song_name+" artist:"+artist_name,limit=5) to restrict to a song name and artist.
# Where the "q" keyword is the query you want to perform on spotify: song_name, artist,...
# while The "limit" keyword will limit the number of returned results.
#
# The second usefull function is:
# .audio_features([URL|URI|ID])
# which returns some 'features of the song', that after cleanup, we can use in order to characterize a song.


# In[218]:


# To make it more “human”, we can randomize the waiting time, thus we dont get blocked by spotify:
from time import sleep
from random import randint
def sleeper():
    for i in range(5):
        print(i)
        wait_time = randint(1,4)
        print("I will sleep for" + str(wait_time) + "seconds.")
        return sleep(wait_time)


# In[219]:


def get_playlist_tracks(username, playlist_id):   #formula rafa para que funcione todo
    results = sp.user_playlist_tracks(username,playlist_id,market="GB")
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
        sleeper()
    return tracks


# In[210]:


playlist1 = get_playlist_tracks("spotify", "37i9dQZF1DWXRqgorJj26U") # fomrula para poner el link de la lista


# In[211]:


rock_list=[]     # hacer lista
for item in range(0,200):
    #print (tracks[item]["track"]["id"])
    rock_list.append(sp.audio_features(playlist1[item]["track"]["id"])[0])


# In[212]:


import pandas as pd   #convertir la lista en dataframe y con columnas
df_1=pd.DataFrame(rock_list)    
df_1=df_1[["id","danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"]]

df_1


# In[ ]:





# In[111]:


playlist2 = get_playlist_tracks("spotify", "4iXi24GdAEXnMUFfdKJ4dh")


# In[114]:


reggeaton_clasicos=[]
for item in range(0,170):
    #print (tracks[item]["track"]["id"])
    reggeaton_clasicos.append(sp.audio_features(playlist2[item]["track"]["id"])[0])


# In[116]:


import pandas as pd
df_2=pd.DataFrame(reggeaton_clasicos)    
df_2=df_2[["id","danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"]]

df_2


# In[117]:


playlist3 = get_playlist_tracks("spotify", "1h0CEZCm6IbFTbxThn6Xcs")


# In[120]:


classical_music=[]
for item in range(0,170):
    #print (tracks[item]["track"]["id"])
    classical_music.append(sp.audio_features(playlist3[item]["track"]["id"])[0])


# In[121]:


#convertir la lista en dataframe y con columnas
df_3=pd.DataFrame(classical_music)    
df_3=df_3[["id","danceability","energy","loudness","speechiness","acousticness",
  "instrumentalness","liveness","valence","tempo","duration_ms"]]

df_3


# In[ ]:


####REVISAR A PARTIR DE AQUI


# In[130]:




playlist4 = get_playlist_tracks("spotify", "3RcRK9HGTAm9eLW1LepWKZ")

hip_hop_90_20=[]
for item in range(0,233):
    #print (tracks[item]["track"]["id"])
    hip_hop_90_20.append(sp.audio_features(playlist4[item]["track"]["id"])[0])
    

  #convertir la lista en dataframe y con columnas
df_4=pd.DataFrame(hip_hop_90_20)    
df_4=df_4[["id","danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"]]

df_4


# In[ ]:





# In[124]:


playlist5 = get_playlist_tracks("spotify", "3RcRK9HGTAm9eLW1LepWKZ")

Pop_Mix=[]
for item in range(0,233):
    #print (tracks[item]["track"]["id"])
    Pop_Mix.append(sp.audio_features(playlist5[item]["track"]["id"])[0])
    

  #convertir la lista en dataframe y con columnas
df_5=pd.DataFrame(Pop_Mix)    
df_5=df_5[["id","danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"]]

df_5


# In[ ]:





# In[126]:


playlist6 = get_playlist_tracks("spotify", "77Ny7ENbK3UM9CvzfSLITH")

indie=[]
for item in range(0,254):
    #print (tracks[item]["track"]["id"])
    indie.append(sp.audio_features(playlist6[item]["track"]["id"])[0])
    

  #convertir la lista en dataframe y con columnas
df_6=pd.DataFrame(indie)    
df_6=df_6[["id","danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"]]

df_6


# In[ ]:





# In[127]:





# In[129]:



playlist7 = get_playlist_tracks("spotify", "3zN691ZuWTKHSRIUyvOIF2")

list_1=[]
for item in range(0,233):
    #print (tracks[item]["track"]["id"])
    list_1.append(sp.audio_features(playlist7[item]["track"]["id"])[0])
    

  #convertir la lista en dataframe y con columnas
df_7=pd.DataFrame(list_1)    
df_7=df_7[["id","danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"]]

df_7


# In[ ]:





# In[407]:



playlist8 = get_playlist_tracks("spotify", "5S8SJdl1BDc0ugpkEvFsIL")

list_2=[]
for item in range(0,1000):
    #print (tracks[item]["track"]["id"])
    list_2.append(sp.audio_features(playlist8[item]["track"]["id"])[0])
    

  #convertir la lista en dataframe y con columnas
df_8=pd.DataFrame(list_2)    
df_8=df_8[["id","danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"]]

df_8


# In[ ]:





# In[ ]:





# In[137]:


playlist9= get_playlist_tracks("spotify", "4Dg0J0ICj9kKTGDyFu0Cv4")

list_3=[]
for item in range(0,1334):
    #print (tracks[item]["track"]["id"])
    list_3.append(sp.audio_features(playlist9[item]["track"]["id"])[0])
    

  #convertir la lista en dataframe y con columnas
df_9=pd.DataFrame(list_3)    
df_9=df_9[["id","danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"]]

df_9


# In[ ]:





# In[213]:


playlist10= get_playlist_tracks("spotify", "77CxoauZX3xv8VsfVwPJxT")

list_4=[]
for item in range(0,1718):
    #print (tracks[item]["track"]["id"])
    list_4.append(sp.audio_features(playlist10[item]["track"]["id"])[0])
    

  #convertir la lista en dataframe y con columnas
df_10=pd.DataFrame(list_4)    
df_10=df_10[["id","danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"]]

df_10


# In[ ]:





# In[ ]:


playlist11= get_playlist_tracks("spotify", "5Q7JPVmkNBFbPF5QVJFLds") 

list_5=[] # si sale error tipo none, hay q anhadir esto, algunas canciones estan como none
for item in range(0,1200):
    #print (tracks[item]["track"]["id"])
    list_5.append(sp.audio_features(playlist11[item]["track"]["id"])[0])
    

  #convertir la lista en


# In[216]:


list_5 = [x for x in list_5 if x is not None]  # si sale error tipo none, hay q anhadir esto, algunas canciones estan como none


# In[217]:


#dataframe y con columnas
df_11=pd.DataFrame(list_5)    
df_11=df_11[["id","danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"]]

df_11


# In[ ]:





# In[ ]:





# In[220]:


playlist12= get_playlist_tracks("spotify", "7B0kSnc5JgV8ljvBAgsKD8")

list_6=[]
for item in range(0,2121):
    #print (tracks[item]["track"]["id"])
    list_6.append(sp.audio_features(playlist12[item]["track"]["id"])[0])
    

  #convertir la lista en dataframe y con columnas
df_12=pd.DataFrame(list_6)    
df_12=df_12[["id","danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"]]

df_12


# In[ ]:





# In[224]:


#list_7 = [x for x in list_7 if x is not None]


# In[245]:


playlist13= get_playlist_tracks("spotify", "2NfTM2df5tHVUquNwet0yB")

list_7=[] 
for item in range(0,2000): #10000
    #print (tracks[item]["track"]["id"])
    list_7 = [x for x in list_7 if x is not None]
    list_7.append(sp.audio_features(playlist13[item]["track"]["id"])[0])
    


# In[ ]:



  #convertir la lista en dataframe y con columnas
df_13=pd.DataFrame(list_7)    
df_13=df_13[["id","danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"]]

df_13


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


playlist14= get_playlist_tracks("spotify", "4RrAuy9EM3YBLZP3K2oXB7")

list_8=[]
for item in range(0,2000): #9965
    #print (tracks[item]["track"]["id"])
    list_8.append(sp.audio_features(playlist14[item]["track"]["id"])[0])
    

  #convertir la lista en dataframe y con columnas
df_14=pd.DataFrame(list_8)    
df_14=df_14[["id","danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"]]

df_14


# In[ ]:





# In[ ]:





# In[241]:


playlist15= get_playlist_tracks("spotify", "4oA5fkQGa2rI7tmGJGT7GT")

list_9=[]
for item in range(0,1500):  #4871 
    #print (tracks[item]["track"]["id"])
    list_9.append(sp.audio_features(playlist15[item]["track"]["id"])[0])
    

  #convertir la lista en dataframe y con columnas
df_15=pd.DataFrame(list_9)    
df_15=df_15[["id","danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"]]

df_15


# In[ ]:





# In[ ]:





# In[ ]:





# In[243]:


final_df = pd.concat([df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_9,df_10,df_11,df_12,df_15]) #concatenar
final_df  #falta el df_8   y 13 14 15


# In[244]:


final_df.drop_duplicates() ## drop duplicaztes


# In[ ]:





# In[ ]:


############## jueves


# In[ ]:


#for today clustering steps:
#1- load the pandas data frame songs of the audio features (minimum 500 songs as diverse as possible))
#2-  standardise the data using standardscaler
#3- save the scaler for future use for the new user input song. save it using pickle
#4- choosing the number of clusters k.
#5- fitting  k means cluster.
#6- build the elbow graph to find the best k.
#7- use the model with the best k to assign every observation in your data frame to its cluster number ( adding cluster columns to the pandas data frame) using model.predict
8#-save the model with the best k as your final model  using pickle.
#for the user input tasks and  the full scenario:
1#- get the song name from the user as an input
2#- play the input song in music embed player.
3#- get the audio features for that song from spotify API. using sp.audio_features(trackid) . pay attention to keep  only the audio features columns.
4#- load the stabdardscaler using pickle and use it to scale the new song.
5#- using mode.predict(new scaled audio record for the new song) to predict the cluster (label) for the new song.
6#- return random song from the same cluster   that the new song belongs to from your data frame and suggest it to the user.
7#-  play it using embedded music player. (edited)


# In[ ]:





# In[356]:


first_copy=final_df


# In[357]:


first_copy


# In[296]:


import numpy as np
import pandas as pd
import pickle
from sklearn import datasets # sklearn comes with some toy datasets to practise
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot
from sklearn.metrics import silhouette_score


# In[359]:


first_copy = first_copy.set_index(["id"]) #paso el ID a index, xq la formula no reconoce los stings, ademas mas tarde tendre que machear el nombre d ela cancion q ponga en el pintu con el id
first_copy


# In[360]:


scaler = StandardScaler()
scaler.fit(first_copy)
X_scaled = scaler.transform(first_copy)
X_scaled_df = pd.DataFrame(X_scaled, columns = first_copy.columns)
display(first_copy.head())
print()
display(X_scaled_df.head())


# In[299]:


#3- save the scaler for future use for the new user input song. save it using pickle


# In[361]:


kmeans = KMeans(n_clusters=3, random_state=1234)
kmeans.fit(X_scaled_df)


# In[362]:


import pickle

#scaler = StandardScaler()
#model = KMeans()

with open("scaler.pickle", "wb") as f:
    pickle.dump(scaler,f)

with open("kmeans_4.pickle", "wb") as f:
    pickle.dump(kmeans,f)


# In[363]:


def load(filename = "filename.pickle"): 
    try: 
        with open(filename, "rb") as f: 
            return pickle.load(f) 
        
    except FileNotFoundError: 
        print("File not found!") 


# In[364]:


scaler2 = load("scaler.pickle")


# In[365]:


#4- choosing the number of clusters k. // #5- fitting  k means cluster. /// 6- build the elbow graph to find the best k.


# In[366]:


K = range(2, 20)   #cambias el numero 50 de este () 
inertia = []       #por disintos n^para ver cual es el breaking -point, ese sera el numeor de clusters q usaras
                   #en este ejerciocio es 9
for k in K:
    print("Training a K-Means model with {} clusters! ".format(k))
    print()
    kmeans = KMeans(n_clusters=k,
                    random_state=1234)
    kmeans.fit(X_scaled_df)
    inertia.append(kmeans.inertia_)

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(16,8))
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('inertia')
plt.xticks(np.arange(min(K), max(K)+1, 1.0))
plt.title('Elbow Method showing the optimal k')


# In[ ]:


#######Silouhette   ver q cluster para usar


# In[367]:


K = range(2, 20)
silhouette = []

for k in K:
    kmeans = KMeans(n_clusters=k,
                    random_state=1234)
    kmeans.fit(X_scaled_df)
    
    filename = "kmeans_" + str(k) + ".pickle"
    with open(filename, "wb") as f:
        pickle.dump(kmeans,f)
    
    silhouette.append(silhouette_score(X_scaled_df, kmeans.predict(X_scaled_df)))


plt.figure(figsize=(16,8))
plt.plot(K, silhouette, 'bx-')
plt.xlabel('k')
plt.ylabel('silhouette score')
plt.xticks(np.arange(min(K), max(K)+1, 1.0))
plt.title('Silhouette Method showing the optimal k')


# In[368]:


kmeans = KMeans(n_clusters=5, random_state=1234)   # decir numero de clusters q quieres
kmeans.fit(X_scaled_df)


# In[369]:


kmeans.labels_


# In[370]:


# assign a cluster to each example
labels = kmeans.predict(X_scaled_df)
# retrieve unique clusters
clusters = np.unique(labels)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = np.where(labels == cluster)
    # create scatter of these samples
    pyplot.scatter(first_copy.to_numpy()[row_ix, 1], first_copy.to_numpy()[row_ix, 3])
    # show the plot
pyplot.show()


# In[371]:


clusters = kmeans.predict(X_scaled_df)
#clusters
pd.Series(clusters).value_counts().sort_index()


# In[372]:


#X_df = pd.DataFrame(X)
first_copy["cluster"] = clusters
first_copy


# In[373]:


#7- use the model with the best k to assign every observation 
#in your data frame to its cluster number ( adding cluster columns to the pandas data frame) using model.predict
first_copy[first_copy['cluster'] == 2].sample()


# In[374]:


#8-save the model with the best k as your final model  using pickle.


# In[375]:


final_first_copy= first_copy


# In[ ]:


#for the user input tasks and  the full scenario:
#1- get the song name from the user as an input
#2- play the input song in music embed player.
#3- get the audio features for that song from spotify API. using sp.audio_features(trackid) . pay attention to keep  only the audio features columns.
#4- load the stabdardscaler using pickle and use it to scale the new song.
#5- using mode.predict(new scaled audio record for the new song) to predict the cluster (label) for the new song.
#6- return random song from the same cluster   that the new song belongs to from your data frame and suggest it to the user.
#7-  play it using embedded music player. (edited)


# In[376]:


final_first_copy.head()


# In[ ]:


##buscar canciones en spotify


# In[408]:


from IPython.display import IFrame    

#track_id = "1rfORa9iYmocEsnnZGMVC4"
track_id= 'spotify:track:3hgl7EQwTutSm6PESsB7gZ'
IFrame(src="https://open.spotify.com/embed/track/"+track_id,
       width="320",
       height="80",
       frameborder="0",
       allowtransparency="true",
       allow="encrypted-media",
      )


# In[412]:





# In[492]:


song = sp.search(q="La Primavera", limit=50,market="GB")  #aqui se escribe el nombre de la cancion
song


# In[493]:


song.keys()


# In[494]:


song["tracks"].keys()


# In[495]:


song["tracks"]["items"]


# In[496]:


song['tracks']['items'][0]['id']


# In[497]:


searched_song=song['tracks']['items'][0]['id']  #automatizo la busqueda
searched_song


# In[498]:


play_song(searched_song)


# In[486]:



def spotify(x):
        
        #Code for searching any song in Spotify.
        userinput = sp.search(q=x,limit=1)
        
        #Getting the ID from the song.
        song_id = userinput['tracks']['items'][0]['id']
        
        #Code for getting the audio features of the given song.
        audio_feautures = sp.audio_features(song_id)   
        
        #Code to transform the new audio features into a DataFrame.
        
        audio_feautures_df = pd.DataFrame(audio_feautures)

        audio_feautures_df_filter = audio_feautures_df[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']]
        
        
        #Using Scaler for the new song.
        x_scaler_audio_feautures = pd.DataFrame(scaler.transform(audio_feautures_df_filter), 
                                                columns=['danceability', 'energy', 'loudness', 
                                                         'speechiness', 'acousticness','instrumentalness', 
                                                         'liveness', 'valence', 'tempo', 'duration_ms'])
    
        #K-means and finding the same cluster.
        cluster_input_song = int(kmeans.predict(x_scaler_audio_feautures))
        
        result = (songs_new[songs_new['cluster']== cluster_input_song].sample())
        
        user_input_id = result['id']
        
        return user_input_id.iloc[0]


# In[487]:


userinput = input('')
id_user_input = spotify(userinput)


# In[ ]:


def play_song(track_id):
    return IFrame(src="https://open.spotify.com/embed/track/"+track_id,
       width="320",
       height="80",
       frameborder="0",
       allowtransparency="true",
       allow="encrypted-media",
      )
play_song(id_user_input)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


import gspread
gc= gspread.service_account(filename = "mythic-delight-370211-09579ff6aae2.json")
gc= gspread.service_account(filename = "mythic-delight-370211-09579ff6aae2.json")

### long import
import sys
get_ipython().system('conda install --yes --prefix {sys.prefix} -c anaconda beautifulsoup4')
get_ipython().system('conda install --yes --prefix {sys.prefix} -c anaconda requests')
####
from bs4 import BeautifulSoup
import requests
import pandas as pd

#####
from bs4 import BeautifulSoup
import requests
import pandas as pd

#check it works

url = "https://docs.google.com/spreadsheets/d/1_mFu_NgmarXt6QbXPFGdWh7LV4xZPpe4pVXDsPCIkBs/edit#gid=0"
response = requests.get(url)
response.status_code # 200 status code means OK!
url
response.content # hacerlo esto

#### long instal (not sure if it must be applied more than once)
#pip install html5lib
#######
from bs4 import BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')
print(soup.title)

######## get data from sheet
# get all the records of the data
records_data = sh.get_values() 

# convert the json to dataframe
records_df = pd.DataFrame.from_dict(records_data)

new_header = records_df.iloc[0] #grab the first row for the header

records_df.columns = new_header #copy the header row as the df header

records_df.drop(index=df.index[0], #remove first row, since now its the header (it was duplicated as row data)
        axis=0, 
        inplace=True)

###### MySQL

#Or you can try to access the database when making the connection:
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="Lupitabonita1010",
  database="mydatabase"
) #If the database does not exist, you will get an error.

# import the module
from sqlalchemy import create_engine
import pymysql

# create sqlalchemy engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",
                               pw="Lupitabonita1010",
                               db="mydatabase"))

# drop table in my sql # IDEA: remove all data and then update new table, always keep the same name, but change the order of the functions
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="Lupitabonita1010",
  database="mydatabase"
)

mycursor = mydb.cursor()

sql = "DROP TABLE mydatabase"

mycursor.execute(sql)

# inserted in MySQL
df_DESDEWEB.to_sql('mydatabase', con = engine, if_exists = 'append', chunksize = 1000) 


# In[ ]:




