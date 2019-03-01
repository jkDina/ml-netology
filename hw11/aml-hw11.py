#Использовать dataset MovieLens
#Построить рекомендации (регрессия, предсказываем оценку) на фичах:
#TF-IDF на тегах и жанрах
#Средние оценки (+ median, variance, etc.) пользователя и фильма
#Оценить RMSE на тестовой выборке

import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

pd.set_option('display.expand_frame_repr', False)

links = pd.read_csv('C:/ml/netology/hw11/ml-latest-small/links.csv')
movies = pd.read_csv('C:/ml/netology/hw11/ml-latest-small/movies.csv')
ratings = pd.read_csv('C:/ml/netology/hw11/ml-latest-small/ratings.csv')
tags = pd.read_csv('C:/ml/netology/hw11/ml-latest-small/tags.csv')



def create_genres_matrix():
    def change_string(s):
        return ' '.join(s.replace(' ', '').replace('-', '').split('|'))

    movie_genres = [change_string(g) for g in movies.genres.values]

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(movie_genres)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    return X_train_tfidf

def create_tags_matrix():
    movies_with_tags = movies.join(tags.set_index('movieId'), on='movieId')
    movies_with_tags.dropna(inplace=True)
    tag_strings = []
    for movie, group in tqdm(movies_with_tags.groupby('title')):
        tag_strings.append(' '.join([str(s).replace(' ', '').replace('-', '') for s in group.tag.values]))
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(tag_strings)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    return X_train_tfidf


def user_predict(user_id, X_train_tfidf, is_tags=False):
    if not is_tags:
        movies_ = movies
    else:
        movies_ = movies[movies['movieId'].isin(tags.movieId)]


    movie_id_to_index = {}
    movie_ids = movies_.movieId

    for index, movie_id in enumerate(movie_ids):
        movie_id_to_index[movie_id] = index
        
    #отбираем строки относящееся к данному user
    user_ratings = ratings[ratings['userId'] == user_id]
    if is_tags:
        user_ratings = user_ratings[user_ratings['movieId'].isin(tags.movieId)]
        

    #отбираем видео, которые оценены данным пользователем
    user_movies = movies_[movies_['movieId'].isin(user_ratings.movieId)]

        
    #получаем индексы фильмов
    movie_indexes = []
    for movie_id in user_movies.movieId:
        movie_indexes.append(movie_id_to_index[movie_id])
    #набор векторов жанров или тегов
    data = []
    target = user_ratings.rating
    #используем строки матрицы для создания векторов фильмов
    for movie_index in movie_indexes:
        row = X_train_tfidf.getrow(movie_index).toarray()[0]
        data.append(row)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print('predictions', predictions)
    print('real', list(y_test))
    print('mean_squared_error = ', mean_squared_error(y_test, predictions))



    
    
    recomend_movies = movies_[~movies_['movieId'].isin(user_ratings.movieId)]
    movie_indexes = []
    for movie_id in recomend_movies.movieId:
        movie_indexes.append(movie_id_to_index[movie_id])

    data = []
    for movie_index in movie_indexes:
        row = X_train_tfidf.getrow(movie_index).toarray()[0]
        data.append(row)

    recomendations = np.argsort(model.predict(data))[::-1][:10]
    print('recomendations = ', recomendations)
    print('movies recomendations = ', movies_.iloc[recomendations])
    
if __name__ == '__main__':
    user_id = 1
    print('predictions using genres:')
    user_predict(user_id, create_genres_matrix())
    print('\n\n')
    print('predictions using tags:')
    user_predict(user_id, create_tags_matrix(), True)
