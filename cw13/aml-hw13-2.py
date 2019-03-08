import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
import matplotlib.pyplot as plt

pd.set_option('display.expand_frame_repr', False)
raw_data = pd.read_table(r'C:/ml/netology/cw13/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv')

raw_data = raw_data.drop(raw_data.columns[1], axis=1)

raw_data.columns = ['user', 'artist', 'plays']

#print(raw_data.head())

data = raw_data.dropna()

#print(data.loc[[1,2]])
#print(data.iloc[np.where(data.plays < 2000)].plays.hist())
#plt.show()

data['user_id'] = data['user'].astype("category").cat.codes
data['artist_id'] = data['artist'].astype("category").cat.codes

#print(data.head())

item_lookup = data[['artist_id', 'artist']].drop_duplicates()
item_lookup['artist_id'] = item_lookup.artist_id.astype(str)

#print(item_lookup.head())

artist_id_name = {}
for index, row in tqdm(item_lookup.iterrows()):
    artist_id_name[row.artist_id] = row.artist

data = data.drop(['user', 'artist'], axis=1)
#print(data.head())

data = data.loc[data.plays != 0]
#print(data.head())

users = list(np.sort(data.user_id.unique()))
artists = list(np.sort(data.artist_id.unique()))
plays = list(data.plays)

print(users[:5])
print(artists[:5])
print(plays[:5])
print(len(users))
print(len(artists))

rows = data.user_id.astype(int)
cols = data.artist_id.astype(int)

data_sparse = sparse.csr_matrix((plays, (cols, rows)), shape=(len(artists), len(users)))

model = AlternatingLeastSquares(factors=50)
model.fit(data_sparse)

userid = 0

user_items = data_sparse.T.tocsr()
recommendations = model.recommend(userid, user_items)

print(recommendations)

for r in recommendations:
    print(artist_id_name[str(r[0])])

itemid = 107209
related = model.similar_items(itemid)

print(related)

for a in related:
    print(artist_id_name[str(a[0])])

    artist_id_name['234786']


    



