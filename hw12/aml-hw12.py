import pandas as pd
import numpy as np
from tqdm import tqdm
from surprise import KNNWithMeans, KNNBasic, KNNWithZScore, KNNBaseline
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
from scipy.spatial.distance import cityblock, cosine, euclidean, hamming, jaccard, rogerstanimoto
data = Dataset.load_builtin('ml-1m')


trainset, testset = train_test_split(data, test_size=.15)

algo = KNNBaseline(k=50, min_k=1, sim_options={'name': 'pearson_baseline', 'user_based': True})
algo.fit(trainset)

test_pred = algo.test(testset)

print('accuracy', accuracy.rmse(test_pred, verbose=True))
print('predict', algo.predict(uid=2, iid='Fight Club (1999)').est)






    



