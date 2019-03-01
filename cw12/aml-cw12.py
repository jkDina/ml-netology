import pandas as pd
import numpy as np
from tqdm import tqdm
from surprise import KNNWithMeans, KNNBasic
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
from scipy.spatial.distance import cityblock, cosine, euclidean, hamming, jaccard, rogerstanimoto


movies = pd.read_csv('C:/ml/netology/cw12/ml-latest-small/movies.csv')
ratings = pd.read_csv('C:/ml/netology/cw12/ml-latest-small/ratings.csv')

pd.set_option('display.expand_frame_repr', False)

movies_with_ratings = movies.join(ratings.set_index('movieId'), on='movieId').reset_index(drop=True)
movies_with_ratings.dropna(inplace=True)

print(movies_with_ratings.head())

num_users = movies_with_ratings.userId.unique().shape[0]

movie_vector ={}

for movie, group in tqdm(movies_with_ratings.groupby('title')):
    movie_vector[movie] = np.zeros(num_users)
    for i in range(len(group.userId.values)):
        u = group.userId.values[i]
        r = group.rating.values[i]
        movie_vector[movie][int(u-1)] = r

print(movie_vector['Toy Story (1995)'])

dataset = pd.DataFrame({
    'uid': movies_with_ratings.userId,
    'iid': movies_with_ratings.title,
    'rating': movies_with_ratings.rating
    })

print('dataset', dataset)

ratings.rating.min()
ratings.rating.max()

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(dataset, reader)

trainset, testset = train_test_split(data, test_size=.15)

algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
algo.fit(trainset)

test_pred = algo.test(testset)

print('accuracy', accuracy.rmse(test_pred, verbose=True))
print('predict', algo.predict(uid=2, iid='Fight Club (1999)').est)




    



