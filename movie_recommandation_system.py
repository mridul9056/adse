#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# In[2]:


movies = pd.read_csv("tmdb_5000_movies.csv")
# Read both split credits files and concatenate them
def load_split_credits():
    credits1 = pd.read_csv("tmdb_5000_credits_1.csv")
    credits2 = pd.read_csv("tmdb_5000_credits_2.csv")
    return pd.concat([credits1, credits2], ignore_index=True)
credits = load_split_credits()

# In[3]:


movies.head(1)

# In[4]:


credits.head(1)

# In[5]:


credits.head(1)['cast'].values

# In[6]:


movies.merge(credits,on='title')

# In[7]:


movies.merge(credits,on='title').shape

# In[8]:


df= movies.merge(credits,on='title')

# In[9]:


df.head(2)

# In[10]:


df.info()

# In[11]:


# generes
# id
# keywords
# title
# overview
# cast
# crew

# these columns are essential and rest of the columns is not essential

# In[12]:


movies = df[['id','title','overview','genres','keywords','cast','crew']]

# In[13]:


movies.head(2)

# In[14]:


movies.isnull().sum()

# In[15]:


movies.dropna(inplace=True)

# In[16]:


movies.isnull().sum()

# In[17]:


movies.duplicated().sum()

# In[18]:


movies.iloc[0].genres

# In[19]:


# into list

# In[20]:


def convert(obj):
    L = []
    for i in obj:
        L.append(i['name'])
    return L

# In[21]:


convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

# In[22]:


import ast
ast.literal_eval

# In[23]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# In[24]:


convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

# In[25]:


movies['genres'].apply(convert)

# In[26]:


movies['genres'] = movies['genres'].apply(convert)

# In[27]:


movies.head()

# In[28]:


# apply on keywords

# In[29]:


movies.iloc[0].keywords

# In[30]:


movies['keywords'].apply(convert)

# In[31]:


movies['keywords'] = movies['keywords'].apply(convert)

# In[32]:


movies.head()

# In[33]:


# we want only top 3 character name of every movie from cast

# In[34]:


movies.iloc[0].cast

# In[35]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

# In[36]:


movies['cast'].apply(convert3)

# In[37]:


movies['cast'] = movies['cast'].apply(convert3)

# In[38]:


movies.head()

# In[39]:


# on crew

# In[40]:


movies.iloc[0].crew

# In[41]:


# we have to fetch the director from crew

# In[42]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# In[43]:


movies['crew'].apply(fetch_director)

# In[44]:


movies['crew'] = movies['crew'].apply(fetch_director)

# In[45]:


movies.head()

# In[46]:


# movies overview is a string  so it has to convert into list
movies['overview'][0]

# In[47]:


movies['overview'].apply(lambda x: x.split())

# In[48]:


movies['overview'] = movies['overview'].apply(lambda x: x.split())

# In[49]:


movies.head()

# In[50]:


# concatinate all the list columns that form a big list and that list convert into paragraph

# In[51]:


# we have to remove the space between the words in all the four columns 
# like 'Sam Worthington' into 'SamWorthington'

# In[52]:


movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])

# In[53]:


movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])

# In[54]:


movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])

# In[55]:


movies.head()

# In[56]:


# all the columns are concatinate into tag column

# In[57]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# In[58]:


movies.head()

# In[59]:


# new dataframe

# In[60]:


new_df = movies[['id','title','tags']]

# In[61]:


new_df

# In[62]:


# covert list into string

# In[63]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

# In[64]:


new_df.head()

# In[65]:


new_df['tags'][0]

# In[66]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

# In[67]:


new_df.head()

# In[68]:


# first movie tags
new_df['tags'][0]

# In[69]:


# sescond movie tags
new_df['tags'][1]

# In[70]:


# find the similarity between the movie tags

# In[71]:


# we need to convert text into vector
# to convert text to vector we use bag of vector
# we take random 5000 most common words and search in every movie tag that how many time that word comes and each word count make
# a vector.

# In[72]:


# we have to remove - in , a , the, etc

# In[73]:


from sklearn.feature_extraction.text import CountVectorizer

# In[74]:


cv = CountVectorizer(max_features=5000,stop_words='english')  # remove stopword english

# In[75]:


vectors = cv.fit_transform(new_df['tags']).toarray()

# In[76]:


vectors

# In[77]:


cv.fit_transform(new_df['tags']).toarray().shape

# In[78]:


vectors[0]

# In[79]:


cv.get_feature_names_out()
['loved','loving','love']
after applying stemming
we get ['love','love','love']
# In[80]:


import nltk

# In[81]:


from nltk.stem.porter import PorterStemmer

# In[82]:


ps = PorterStemmer()

# In[83]:


# for ecxample 
ps.stem('loved')

# In[84]:


ps.stem('loving')

# In[85]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)

# In[86]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')

# ### original string 
# 
# 'in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron'

# In[87]:


new_df['tags'] = new_df['tags'].apply(stem)

# In[88]:


from sklearn.feature_extraction.text import CountVectorizer

# In[89]:


cv = CountVectorizer(max_features=5000,stop_words='english')  # remove stopword english

# In[90]:


vectors = cv.fit_transform(new_df['tags']).toarray()

# In[91]:


vectors

# In[92]:


cv.get_feature_names_out()

# In[95]:


# calculate distance between each movie with every movie (calculate vector distance)

# In[96]:


from sklearn.metrics.pairwise import cosine_similarity

# In[98]:


cosine_similarity(vectors)

# In[99]:


cosine_similarity(vectors).shape

# In[100]:


similarity = cosine_similarity(vectors)

# In[103]:


similarity[0]  # this tells us that 1 movies similarity score with each movie upto 4806 for example 1 movie similar with 1 movie
               # is equal to 1 therefore we got 1, and the 1 movie is similar with 2nd movie = 0.08346223 amd so on......

# In[117]:


# find similar  movie using sorting 
sorted(similarity[0],reverse=True)

# In[118]:


# we want starting five movie

# In[104]:


# similarity check for second movie 
similarity[1]

# ##### we are going to make a function where we give a movie name and they give five similer movies name

# In[105]:




# In[109]:


new_df[new_df['title'] == 'Avatar']

# In[112]:


# index of my movie
new_df[new_df['title'] == 'Avatar'].index[0]

# In[113]:


new_df[new_df['title'] == 'Batman Begins'].index[0]

# In[114]:


def recommand(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    return

# In[ ]:



