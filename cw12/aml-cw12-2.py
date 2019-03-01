#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from surprise import KNNWithMeans, KNNBasic
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split

import pandas as pd


# In[ ]:


movies = pd.read_csv('../lecture-1/movies.csv')
ratings = pd.read_csv('../lecture-1/ratings.csv')


# In[ ]:


ratings.head()


# In[ ]:


movies_with_ratings = movies.join(ratings.set_index('movieId'), on='movieId').reset_index(drop=True)
movies_with_ratings.dropna(inplace=True)


# In[ ]:


movies_with_ratings[movies_with_ratings.userId == 2.0].title.unique()


# In[ ]:


dataset = pd.DataFrame({
    'uid': movies_with_ratings.userId,
    'iid': movies_with_ratings.title,
    'rating': movies_with_ratings.rating
})


# In[ ]:


dataset.head()


# In[ ]:


ratings.rating.min()


# In[ ]:


ratings.rating.max()


# In[ ]:


reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(dataset, reader)


# In[ ]:


trainset, testset = train_test_split(data, test_size=.15)


# In[ ]:


algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
algo.fit(trainset)


# In[ ]:


test_pred = algo.test(testset)


# In[ ]:


accuracy.rmse(test_pred, verbose=True)


# In[ ]:


algo.predict(uid=2, iid='Fight Club (1999)').est

