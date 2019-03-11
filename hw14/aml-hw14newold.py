#Для тех кто не смотрел ничего - TopRecommender
#Для тех, кто смотрел менее ___ фильмов content based
#Кто много смотрит ______ ________

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.neighbors import NearestNeighbors

import pandas as pd
import numpy as np

from collections import Counter
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix


links = pd.read_csv('links.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')

pd.set_option('display.expand_frame_repr', False)

print(links.head())
print(movies.head())
print(ratings.head())
print(tags.head())

user_id = 2

#отбираем строки относящееся к данному user в dataframe ratings
user_ratings = ratings[ratings['userId'] == user_id]
    
#отбираем видео, которые оценены данным пользователем
user_movies = movies[movies['movieId'].isin(user_ratings.movieId)]

amount = user_movies.shape[0]
print('>>>>>>>>>>>>>>>>>>>>>>>>>>> amount = ', amount)



if amount <= 5:
    #TopRecommender
    class TopRecommender(object):
        def fit(self, train_data):
            count = Counter(train_data['movieId'])
            self.predictions = count.most_common()
        def predict(self, user_id, n_recommendations=10):
            return self.predictions[:n_recommendations]

    time_range = sorted(ratings['timestamp'])
    border_timestamp = time_range[int(len(time_range) * 0.85)]

    X_train = ratings[ratings.timestamp < border_timestamp]
    X_test = ratings[ratings.timestamp >= border_timestamp]
    #print(X_train)

    model = TopRecommender()
    model.fit(X_train)
    recoms = model.predict(user_id)

    print('recoms = ', recoms)
    for rec in recoms:
        print(movies[movies.movieId == rec[0]].iloc[0].title)

elif 5 < amount <= 10:
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


    for index, movie_id in enumerate(movie_ids):
        movie_id_to_index[movie_id] = index
    #print('movie_id_to_index = ', movie_id_to_index)

        
    #получаем индексы фильмов
    movie_indexes = []

    for movie_id in user_movies.movieId:
        movie_indexes.append(movie_id_to_index[movie_id])

    #набор векторов жанров
    data = []
    target = user_ratings.rating
        
    #используем строки матрицы для создания векторов фильмов(жанров)
    for movie_index in movie_indexes:
        print('X_train_tfidf=', X_train_tfidf)
        row = X_train_tfidf.getrow(movie_index).toarray()[0]
        data.append(row)


    print('data = ', data)
    print('target = ', target)
         
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    
    for index in reversed(np.argsort(predictions)):
        row = X_test[index]
        for i, r in enumerate(data):
            if r is row:
                index_ = movie_indexes[i]
                print(movies.iloc[index_].title)
                continue
    #print('predictions', predictions)
    #print('real', list(y_test))
    #print('mean_squared_error = ', mean_squared_error(y_test, predictions))
else:
    print('ratings = ', ratings.head())
    n_components = 30
    rows = ratings.userId.apply(lambda userId: userId-1)
    cols = ratings.movieId.apply(lambda movieId: movieId-1)
    vals = ratings.rating
 
    interactions_matrix = csr_matrix((vals, (rows, cols)))
    model = TruncatedSVD(n_components = n_components, algorithm='arpack')
    model.fit(interactions_matrix)

    user_interactions = interactions_matrix.getrow(user_id - 1)
    user_low_dimensions = model.transform(user_interactions)

    print('user_low_dimensions = ', user_low_dimensions)
    user_predictions = model.inverse_transform(user_low_dimensions)[0]
    recommendations = []
    print(user_predictions)  

    max_n = 10
    #Пробегаем по колонкам в порядке убывания предсказанного значения
    for movie_idx in reversed(np.argsort(user_predictions)):
        #Добавляем фильм к рекомендациям только если пользователь его еще не смотрел
        if user_interactions[0, movie_idx] == 0.0:
            movie_id = movie_idx + 1
            score = user_predictions[movie_idx]
            recommendations.append((movie_id, score))
            
            if (len(recommendations) >= max_n):
                break
        
    print('recommendations = ', recommendations)
    for rec in recommendations:
        print(movies[movies.movieId == rec[0]].iloc[0].title)
    
    
    
    
