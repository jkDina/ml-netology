#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from datetime import datetime

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


links = pd.read_csv('../lecure-1/links.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')


# In[ ]:


tags.head()


# In[ ]:


tags.groupby('userId').tag.count().hist(bins=30)


# In[ ]:


tags.groupby('userId').tag.count().median()


# In[ ]:


tags.groupby('userId').tag.count().mean()


# In[ ]:


tags.groupby('movieId').tag.count().hist(bins=30)


# In[ ]:


tags.groupby('movieId').tag.count().mean()


# In[ ]:


tags.groupby('movieId').tag.count().median()


# In[ ]:


year_month = []

for t in tqdm_notebook(tags.timestamp.values):
    d = datetime.fromtimestamp(t)
    year_month.append(str(d.year) + '-' + str(d.month))


# In[ ]:


tags['year_month'] = np.array(year_month)


# In[ ]:


tags.year_month.value_counts()[:30].plot.bar()


# In[ ]:


tags.groupby('year_month').tag.count().hist(bins=30)


# In[ ]:


tags.groupby('year_month').tag.count().mean()


# In[ ]:


tags.groupby('year_month').tag.count().median()


# In[ ]:


num_genres_on_movie = [len(g.split('|')) for g in movies.genres.values]


# In[ ]:


plt.hist(num_genres_on_movie)


# In[ ]:


np.mean(num_genres_on_movie)


# In[ ]:


np.median(num_genres_on_movie)

