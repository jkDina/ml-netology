'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
x_moons, y_moons = make_moons(n_samples=200, noise=0.1)
plt.figure(figsize=(12,9))
plt.scatter(x_moons[:,0], x_moons[:,1], c=y_moons)
plt.show()'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
x_blobs, y_blobs = make_blobs(n_samples=200, centers=7)
'''plt.figure(figsize=(12,9))
plt.scatter(x_blobs[:,0], x_blobs[:,1], c=y_blobs)
plt.show()'''

def distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def MyKMeans(X, n_clusters, as_generator=False):
    centroids = X[np.random.choice(list(range(len(X))), n_clusters)]
    n_epochs = 100
    for _ in range(n_epochs):
        #считаем расстояние до кластеров
        belonging = []
        for sample in X:
            distances = []
            for center in centroids:
                distances.append(distance(sample,center))
            belonging.append(distances)
        #определяем принадлежность кластерам
        belonging=np.array(belonging)
        belonging=np.argmin(belonging, axis=1)

        #пересчитываем центры
        new_centroids = []
        for c in range(n_clusters):
            points = X[belonging==c]
            new_centroids.append(points.mean(axis=0))

        centroids = np.array(new_centroids)

        if as_generator:
            yield belonging, centroids

    yield belonging, centroids

generator_blobs= MyKMeans(x_blobs, 7)
y_pred, centers = next(generator_blobs)
plt.figure(figsize=(12,9))
plt.scatter(x_blobs[:,0], x_blobs[:,1], c=y_pred)
plt.scatter(centers[:,0], centers[:,1], c='black', s=300)
plt.show()

      
        
                


