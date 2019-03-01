#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

from tqdm import tqdm_notebook

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.neighbors import NearestNeighbors

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


links = pd.read_csv('../lecture-1/links.csv')
movies = pd.read_csv('../lecture-1/movies.csv')
ratings = pd.read_csv('../lecture-1/ratings.csv')
tags = pd.read_csv('../lecture-1/tags.csv')


# In[ ]:


movies.head(10)


# In[ ]:


def change_string(s):
    return ' '.join(s.replace(' ', '').replace('-', '').split('|'))


# In[ ]:


movie_genres = [change_string(g) for g in movies.genres.values]


# In[ ]:


movie_genres[:10]


# In[ ]:


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(movie_genres)


# In[ ]:


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# In[ ]:


neigh = NearestNeighbors(n_neighbors=7, n_jobs=-1, metric='euclidean') 
neigh.fit(X_train_tfidf)


# In[ ]:


test = change_string("Adventure|Comedy|Fantasy|Crime")

predict = count_vect.transform([test])
X_tfidf2 = tfidf_transformer.transform(predict)

res = neigh.kneighbors(X_tfidf2, return_distance=True)


# In[ ]:


]:
res


# In[ ]:


movies.iloc[res[1][0]]


# In[ ]:


movies.head()


# In[ ]:


tags.head()


# In[ ]:


movies_with_tags = movies.join(tags.set_index('movieId'), on='movieId')


# In[ ]:


movies_with_tags.head()


# In[ ]:


movies_with_tags[movies_with_tags.title == 'Toy Story (1995)']


# In[ ]:


movies_with_tags.tag.unique()


# In[ ]:


movies_with_tags.dropna(inplace=True)


# In[ ]:


movies_with_tags.title.unique().shape


# In[ ]:


tag_strings = []
movies = []

for movie, group in tqdm_notebook(movies_with_tags.groupby('title')):
    tag_strings.append(' '.join([str(s).replace(' ', '').replace('-', '') for s in group.tag.values]))
    movies.append(movie)


# In[ ]:


tag_strings[:5]


# In[ ]:


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(tag_strings)


# In[ ]:


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# In[ ]:


neigh = NearestNeighbors(n_neighbors=10, n_jobs=-1, metric='manhattan') 
neigh.fit(X_train_tfidf)


# In[ ]:


for i in range(len(movies)):
    if 'Magnolia (1999)' == movies[i]:
        print(i)


# In[ ]:


tag_strings[822]


# In[ ]:


test = change_string('pixar pixar fun')

predict = count_vect.transform([test])
X_tfidf2 = tfidf_transformer.transform(predict)

res = neigh.kneighbors(X_tfidf2, return_distance=True)


# In[ ]:


res


# In[ ]:


for i in res[1][0]:
    print(movies[i])

