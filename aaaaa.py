#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


movies = pd.read_csv('datasheet.csv')
credits = pd.read_csv('datasheet1.csv') 


# In[3]:


movies.head()


# In[4]:


movies.head(1)


# In[5]:


movies.shape


# In[6]:


credits.head()


# In[7]:


movies = movies.merge(credits,on='title')


# In[8]:


movies.head(1)


# In[9]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[10]:


movies.head()


# In[11]:


movies.isnull().sum()


# In[12]:


movies.duplicated().sum()


# In[13]:


movies.iloc[0].genres


# In[14]:


import ast


# In[15]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[16]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[17]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[18]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[19]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[20]:


movies['cast'].apply(convert3)
movies.head()


# In[21]:


movies['crew'][0]


# In[ ]:





# In[22]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[23]:


movies['crew'].apply(fetch_director)


# In[24]:


movies['crew'] = movies['crew'].apply(fetch_director)
movies.head()


# In[25]:


movies['overview'][0]


# In[26]:


movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
movies.sample(5)


# In[27]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace("","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace("","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace("","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace("","")for i in x])


# In[43]:


movies.head()


# In[35]:


movies['tags'] = (
    movies['overview'].astype(str) +
    movies['genres'].apply(lambda x: ' '.join(map(str, x))).astype(str) +
    movies['keywords'].apply(lambda x: ' '.join(map(str, x))).astype(str) +
    movies['cast'].apply(lambda x: ' '.join(map(str, x))).astype(str) +
    movies['crew'].apply(lambda x: ' '.join(map(str, x))).astype(str)
)


# In[44]:


new=movies[['movie_id','title','tags']]
new.head()


# In[46]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
    


# In[47]:


cv


# In[49]:


vector = cv.fit_transform(new['tags'].values.astype('U')).toarray()


# In[50]:


vector.shape


# In[51]:


from sklearn.metrics.pairwise import cosine_similarity


# In[52]:


similarity = cosine_similarity(vector)


# In[53]:


similarity


# In[54]:


new[new['title'] == 'The Lego Movie'].index[0]


# In[55]:


distance= sorted(list(enumerate(similarity[2])) ,reverse=True,key=lambda vector:vector[1])
for i in distance[0:5]:
    print(new.iloc[i[0]].title)


# In[57]:


def recommend(movies):
    index = new[new['title'] == movies].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)
        
    


# In[58]:


recommend('Gandhi')


# In[59]:


import pickle


# In[60]:


pickle.dump(new,open('movie_list.pkl','wb'))


# In[61]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[62]:


pickle.load(open('movie_list.pkl','rb'))


# In[ ]:




