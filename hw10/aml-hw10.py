import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

pd.set_option("display.expand_frame_repr", False)
links = pd.read_csv('links.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')

#print('LINKS')
#print(links.head())
#print('MOVIES')
#print(movies.head())
#print('RATINGS')
#print(ratings.head())
print('TAGS')
print(tags.head())


#соединим таблицу с рейтингами и названиями фильмов
joined_ratings = ratings.join(movies.set_index('movieId'), on='movieId')

#print('JOINED_RATINGS')
#print(joined_ratings.head())

#посмотрим на гистограмму распределения оценок
joined_ratings.rating.hist()
#plt.show()

#гистограмма по количеству оценок на фильм
joined_ratings.groupby('title').rating.count().hist()
#plt.show()

#достанем топ фильмов по оценкам
top_films = joined_ratings.groupby('title')[['rating']].mean().sort_values('rating', ascending=False)
#print(top_films.head(10))


#возьмем только фильмы с наивысшей средней оценкой в 5.0
films_with_highest_marks = top_films.iloc[np.where(top_films.rating == 5.0)].index
#print(films_with_highest_marks)
#print('---------')

#возьмем только фильмы с наивысшей средней оценкой в 5.0
#films_with_highest_marks = top_films.iloc[np.where(top_films.rating == 5.0)]
#print(films_with_highest_marks)

#достанем по каждому фильму количество рейтингов
title_num_ratings ={}
for title, group in tqdm(joined_ratings.groupby('title')):
    title_num_ratings[title] = group.userId.unique().shape[0]
#print(list(title_num_ratings.items())[0:10])

# выведем топ фильмов со средней оценкой в 5.0
#по количеству отзывов
#и увидим, что рейтинг получается не самый удачный

top_rating = sorted([(title_num_ratings[f], f) for f in films_with_highest_marks], key=lambda x: x[0], reverse = True)[:10]
#print(top_rating)

#Приняли решение сортировать фильмы по следующей метрике: средняя оценка фильма,
#умноженная на нормированное количество рейтингов
#достанем простые статистики по количеству рейтингов
min_num_ratings = np.min([title_num_ratings[f] for f in title_num_ratings.keys()])
max_num_ratings = np.max([title_num_ratings[f] for f in title_num_ratings.keys()])
mean_num_ratings = np.mean([title_num_ratings[f] for f in title_num_ratings.keys()])
median_num_ratings = np.median([title_num_ratings[f] for f in title_num_ratings.keys()])
#print('min_num_ratings', min_num_ratings)
#print('max_num_ratings', max_num_ratings)
#print('mean_num_ratings', mean_num_ratings)
#print('median_num_ratings', median_num_ratings)

#считаем средний рейтинг на каждый фильм
title_mean_rating = {}
for title, group in tqdm(joined_ratings.groupby('title')):
    title_mean_rating[title] = group.rating.mean()
#print(list(title_mean_rating.items())[0:10])

#посчитаем нашу метрику для каждого фильма из датасета
film_with_our_mark = []
for f in title_num_ratings.keys():
    film_with_our_mark.append(
        (f, title_mean_rating[f] * (title_num_ratings[f] - mean_num_ratings) / (max_num_ratings - min_num_ratings))
    )
#выводим топ 20 и получилось уже очень неплохо
#print(list(sorted(film_with_our_mark, key=lambda x: x[1], reverse=True))[:20])

#Появилась гипотеза использовать теги в ранжировании фильмов,
#решили считать не только количество отзывов, а ещё и количество проставленных тегов на фильм

#соединим уже созданную таблицу с таблицей с проставленными тегами по фильмам
joined_with_tags = joined_ratings.join(tags.set_index('movieId'), on='movieId', lsuffix='_left', rsuffix='_right')
#print(joined_with_tags.head())

#достанем по каждому фильму количество рейтингов
title_num_actions = {}
for title, group in tqdm(joined_with_tags.groupby('title')):
    title_num_actions[title] = group.shape[0]
#print(list(title_num_actions.items())[0:10])

min_num_actions = np.min([title_num_actions[f] for f in title_num_ratings.keys()])
max_num_actions = np.max([title_num_actions[f] for f in title_num_ratings.keys()])
mean_num_actions = np.mean([title_num_actions[f] for f in title_num_ratings.keys()])
median_num_actions = np.median([title_num_actions[f] for f in title_num_ratings.keys()])
#print('min_num_actions', min_num_actions)
#print('max_num_actions', max_num_actions)
#print('mean_num_actions', mean_num_actions)
#print('median_num_actions', median_num_actions)

film_with_new_mark = []
for f in title_num_actions.keys():
    #посчитаем нашу новую метрику для каждого фильма из датасета
    film_with_new_mark.append(
        (f, title_mean_rating[f] * (title_num_actions[f] - mean_num_actions) / (max_num_ratings - min_num_ratings))
    )
#выведем топ фильмов по новой метрике
#print(list(sorted(film_with_new_mark, key=lambda x: x[1], reverse=True))[:20])

    


    

























