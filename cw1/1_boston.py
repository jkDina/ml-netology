#!/usr/bin/env python
# coding: utf-8

# # 1. Предсказание цены на недвижимость

# In[3]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
from pylab import rcParams
rcParams['figure.figsize'] = (9, 6)


# In[4]:


import numpy as np


# ### Данные

# в sklearn есть модуль, содержащий набор классических датасетов, воспользуемся им:

# In[5]:


from sklearn.datasets import load_boston


# In[6]:


print(load_boston()['DESCR'])


# In[7]:


X, y = load_boston(return_X_y = True)


# In[8]:


X.shape


# In[9]:


y.shape


# ### Формирование выборок

# разделим данные на 2 части, обучающую и тренировочную выборки:
# 1. фиксируем размер обучающей выборки
# 2. выделяем подмассивы данных из X, y

# In[10]:


"""
make X_train, X_test, y_train, y_test
"""


# ### Построение регрессии и предсказания по тестовой выборке

# In[11]:


"""
make y_pred
"""


# ### Оценка

# In[12]:


"""
make visual comparement 
"""


# ### MSE

# MSE - среднеквардратичная ошибка, т.е. среднее значение суммы квадратов ошибок

# In[13]:


"""
count MSE
"""

