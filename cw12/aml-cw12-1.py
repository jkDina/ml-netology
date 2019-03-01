#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

from tqdm import tqdm_notebook


# In[ ]:


movies = pd.read_csv('../lecture-1/movies.csv')
ratings = pd.read_csv('../lecture-1/ratings.csv')


# In[ ]:


movies_with_ratings = movies.join(ratings.set_index('movieId'), on='movieId').reset_index(drop=True)
movies_with_ratings.dropna(inplace=True)


# In[ ]:


movies_with_ratings.head()


# In[ ]:


num_users = movies_with_ratings.userId.unique().shape[0]


# In[ ]:


movie_vector = {}

for movie, group in tqdm_notebook(movies_with_ratings.groupby('title')):
    movie_vector[movie] = np.zeros(num_users)
    
    for i in range(len(group.userId.values)):
        u = group.userId.values[i]
        r = group.rating.values[i]
        movie_vector[movie][int(u - 1)] = r


# In[ ]:


movie_vector['Toy Story (1995)']


# In[ ]:


from scipy.spatial.distance import cityblock, cosine, euclidean, hamming, jaccard, rogerstanimoto


# In[ ]:




