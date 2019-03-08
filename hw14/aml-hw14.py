#Для тех кто не смотрел ничего - TopRecommender
#Для тех, кто смотрел менее ___ фильмов content based
#Кто много смотрит метод ______   потом _________
from surprise import SVD, SVDpp
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
#from surprise.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.neighbors import NearestNeighbors

import pandas as pd
import numpy as np

from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from collections import Counter

links = pd.read_csv('links.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')

pd.set_option('display.expand_frame_repr', False)

print(links.head())
print(movies.head())
print(ratings.head())
print(tags.head())


#TopRecommender
class TopRecommender(object):
    def fit(self, train_data):
        count=Counter(train_data['movieId'])
        self.predictions = count.most_common()
    def predict(self, user_id, n_recommendations=10):
        return self.predictions[:n_recommendations]
#model=TopRecommender()
#model.fit()
#model.predict()


#ContentBased
def change_string(s):
    return ' '.join(s.replace(' ', '').replace('-', '').split('|'))

movie_genres = [change_string(g) for g in movies.genres.values]

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(movie_genres)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

movie_id_to_index = {}

movie_ids = movies.movieId

user_id = 2

for index, movie_id in enumerate(movie_ids):
    movie_id_to_index[movie_id] = index
#print('movie_id_to_index = ', movie_id_to_index)
#отбираем строки относящееся к данному user в dataframe ratings
user_ratings = ratings[ratings['userId'] == user_id]
    
#отбираем видео, которые оценены данным пользователем
user_movies = movies[movies['movieId'].isin(user_ratings.movieId)]
    
#получаем индексы фильмов
movie_indexes = []

for movie_id in user_movies.movieId:
    movie_indexes.append(movie_id_to_index[movie_id])

#набор векторов жанров
data = []
target = user_ratings.rating
    
#используем строки матрицы для создания векторов фильмов
for movie_index in movie_indexes:
    row = X_train_tfidf.getrow(movie_index).toarray()[0]
    data.append(row)


print('data = ', data)
print('target = ', target)
     
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print('predictions', predictions)
print('real', list(y_test))
print('mean_squared_error = ', mean_squared_error(y_test, predictions))



from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

model = LightFM()
model.fit(X_train, epochs=10)

    

