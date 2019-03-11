#Для тех кто не смотрел ничего - TopRecommender
#Для тех, кто смотрел от 5 до 20 фильмов рекомендации на основе жанров
#У кого много оценок - SVD + tags
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.neighbors import NearestNeighbors

import pandas as pd
import numpy as np

from collections import defaultdict, Counter
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
print('user_ratings = \n', user_ratings)
    
#отбираем видео, которые оценены данным пользователем
user_movies = movies[movies['movieId'].isin(user_ratings.movieId)]

amount = user_movies.shape[0]
print('amount = ', amount)

if amount <= 5:
    #TopRecommender
    class TopRecommender(object):
        def fit(self, train_data):
            count = Counter(train_data['movieId'])
            self.predictions = count.most_common()
        def predict(self, user_id, n_recommendations=10):
            return self.predictions[:n_recommendations]

    model = TopRecommender()
    model.fit(ratings)
    recoms = model.predict(user_id)

    for rec in recoms:
        print(movies[movies.movieId == rec[0]].iloc[0].title)

elif 5 < amount <= 20:
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

        
    #получаем индексы фильмов
    movie_indexes = []

    for movie_id in user_movies.movieId:
        movie_indexes.append(movie_id_to_index[movie_id])

    #набор векторов жанров
    data = []
    target = user_ratings.rating
        
    #используем строки матрицы для создания векторов фильмов
    #i = 0
    data_to_index = {}
    for movie_index in movie_indexes:
        row = X_train_tfidf.getrow(movie_index).toarray()[0]
        data.append(row)
         
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('mean_squared_error = ', mean_squared_error(y_test, predictions))

    #предсказываем оценки для еще не просмотренных фильмов
    new_movies = movies[~movies.movieId.isin(user_movies.movieId)]
    new_movie_ids = []
    for movie_id in new_movies.movieId.values:
        movie_index = movie_id_to_index[movie_id]
        row = X_train_tfidf.getrow(movie_index).toarray()[0]
        new_movie_ids.append(row)

    scores = model.predict(new_movie_ids)
    #рекомендуем 20 фильмов
    for index in list(reversed(np.argsort(scores)))[:20]:
        print(new_movies.iloc[index].title)
else:
    #рекомендации на основе SVD разложения
    n_components = 30
    rows = ratings.userId.apply(lambda userId: userId-1)
    cols = ratings.movieId.apply(lambda movieId: movieId-1)
    vals = ratings.rating
 
    interactions_matrix = csr_matrix((vals, (rows, cols)))
    model = TruncatedSVD(n_components = n_components, algorithm='arpack')
    model.fit(interactions_matrix)

    user_interactions = interactions_matrix.getrow(user_id - 1)
    user_low_dimensions = model.transform(user_interactions)

    user_predictions = model.inverse_transform(user_low_dimensions)[0]
    recommendations = []

    max_n = 200
    #Пробегаем по колонкам в порядке убывания предсказанного значения
    for movie_idx in reversed(np.argsort(user_predictions)):
        #Добавляем фильм к рекомендациям только если пользователь его еще не смотрел
        if user_interactions[0, movie_idx] == 0.0:
            movie_id = movie_idx + 1
            score = user_predictions[movie_idx]
            recommendations.append((movie_id, score))
            
            if (len(recommendations) >= max_n):
                break
        
    #print('recommendations = ', recommendations)
    #for rec in recommendations:
        #print(movies[movies.movieId == rec[0]].iloc[0].title)
      
    #добавим учет тегов 
    movies_with_tags = movies.join(tags.set_index('movieId'), on='movieId')
    movies_with_tags.dropna(inplace=True)
    tag_strings = []
    for movie, group in tqdm(movies_with_tags.groupby('title')):
        tag_strings.append(' '.join([str(s).replace(' ', '').replace('-', '') for s in group.tag.values]))

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(tag_strings)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    movie_id_to_index = {}
    movie_ids = movies_with_tags.movieId.unique()

    for index, movie_id in enumerate(movie_ids):
        movie_id_to_index[movie_id] = index
        
    #отбираем строки относящееся к данному user
    user_ratings = ratings[ratings['userId'] == user_id]
    user_ratings = user_ratings[user_ratings['movieId'].isin(tags.movieId)]

    movies_ = movies[movies['movieId'].isin(tags.movieId)]
    user_movies = movies_[movies_['movieId'].isin(user_ratings.movieId)]
    
    #получаем индексы фильмов
    #movie_indexes = np.arange(len(movie_ids))
    movie_indexes = []
    for movie_id in user_movies.movieId:
        movie_indexes.append(movie_id_to_index[movie_id])

    #набор векторов тегов
    data = []
    target = user_ratings.rating
    
    #используем строки матрицы для создания векторов фильмов
    for movie_index in movie_indexes:
        row = X_train_tfidf.getrow(movie_index).toarray()[0]
        data.append(row)
        
    model2 = RandomForestRegressor()
    model2.fit(data, target)

    #добавим к результатам предсказаний по SVD то, что получили по тегам
    pred_data = []
    for rec in recommendations:
        movieId = rec[0]
        if movieId in movie_ids:
            pred_data.append(X_train_tfidf.getrow(movie_id_to_index[movieId]).toarray()[0])
        else:
            pred_data.append([0] * X_train_tfidf.shape[1])

    predictions2 = model2.predict(pred_data)
    predictions2 = predictions2 / (np.max(predictions2) + np.max([x[1] for x in recommendations]))

    #к score из SVD прибавим score по тегам
    result_recommendations = np.array([x[1] for x in recommendations]) + predictions2

    #print('recommendations = ', recommendations)
    #print('predictions2 = ', predictions2)
    #print('result_recommendations = ', result_recommendations)
    for i, rec in enumerate(sorted(result_recommendations, reverse=True)[:20]):
        print(movies[movies.movieId == recommendations[i][0]].iloc[0].title)
    
    
    
