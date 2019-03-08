from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split

import pandas as pd

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

#print(ratings.head())

movies_with_ratings = movies.join(ratings.set_index('movieId'), on='movieId')
movies_with_ratings.dropna(inplace=True)

dataset = pd.DataFrame({
    'uid': movies_with_ratings.userId,
    'iid': movies_with_ratings.title,
    'rating': movies_with_ratings.rating
    })

#print(dataset.head(30))
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(dataset,reader)
#print(data.df.head(30))

#print(list(zip(dataset.rating.head(100), data.df.rating.head(100))))

trainset,testset = train_test_split(data, test_size=.15, random_state=42)

algo = SVD(n_factors=20, n_epochs=20)
algo.fit(trainset)
test_pred = algo.test(testset)

print('rmse = ', accuracy.rmse(test_pred, verbose=True))
print('prediction = ', algo.predict(uid=5.0, iid='MortalKombat(1995)'))




